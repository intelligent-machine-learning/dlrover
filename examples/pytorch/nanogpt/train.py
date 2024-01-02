# Copyright 2023 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The start command on a local ndoe:

dlrover-run --nproc_per_node=2 train.py \
    --n_layer 48 --n_head 16 --n_embd 1600 --data_dir './' \
    --epochs 50 --save_memory_interval 50 --save_storage_interval 500
"""


import argparse
import contextlib
import os
import time

import torch
from lora import apply_lora
from torch.nn.parallel import DistributedDataParallel as DDP
from train_utils import (
    add_train_args,
    cleanup,
    create_lora_config,
    get_data_loaders,
    get_lr,
    gpt_init,
    log_rank0,
    setup,
)

from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler
from dlrover.trainer.torch.elastic.trainer import ElasticTrainer
from dlrover.trainer.torch.flash_checkpoint.ddp import (
    DdpCheckpointer,
    StorageType,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# We should use a shared storage to persist the checkpiont.
checkpoint_dir = "/nas/nanogpt-ckpt/"


def train():
    args = arg_parser()
    setup()
    os.makedirs(checkpoint_dir, exist_ok=True)
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.batch_size
    if gradient_accumulation_steps == 0:
        gradient_accumulation_steps = world_size
    assert gradient_accumulation_steps % world_size == 0
    block_size = args.block_size
    gradient_accumulation_steps //= world_size
    tokens_per_iter = (
        gradient_accumulation_steps * world_size * batch_size * block_size
    )  # noqa: E501
    log_rank0(f"tokens per iteration will be: {tokens_per_iter:,}")
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(device)
    # Note: float16 data type will automatically use a GradScaler
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )
    # Auto implement a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        contextlib.nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    train_loader, val_loader, meta_vocab_size = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        block_size=block_size,
    )
    model = gpt_init(meta_vocab_size, args=args)
    lora_config = create_lora_config(args)
    if lora_config is not None:
        log_rank0(f"apply lora config {lora_config}")
        apply_lora(model, **lora_config)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    model = model.to(device)
    # Device
    if torch.cuda.is_available() and device_type == "cuda":
        # Create model and move it to GPU with id rank
        model = model.to(local_rank)
        print(f"Running basic DDP example on local rank {local_rank}.")
        model = DDP(model, device_ids=[local_rank])
        print(f"Model device {model.device}")
    else:
        print(f"Running basic CPU example on device {device}.")
        model = model.to(device)
        model = DDP(model)
        print(f"Model device {model.device}")
    # Optimizer
    log_rank0(f"creating optimizer...{model.parameters()}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )

    # Compile the model
    if compile == "True":
        log_rank0("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    # Training loop
    total_time = 0.0
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    raw_model = model.module  # Unwrap DDP/FSDP container if needed
    running_mfu = -1.0
    iter_num = 0
    decay_lr = args.decay_lr
    max_iters = args.max_iters
    log_interval = args.log_interval
    grad_clip = args.grad_clip
    learning_rate = args.learning_rate
    elastic_trainer = ElasticTrainer(
        model=model,
        dataloader=train_loader,
    )
    optimizer = elastic_trainer.prepare(optimizer)

    # Forward backward update, with optional gradient accumulation
    # to simulate larger batch size and using the GradScaler
    # if data type is float16

    checkpointer = DdpCheckpointer(checkpoint_dir)

    t0 = time.time()
    ckpt_dict = {}
    if args.use_native_ckpt:
        ckpt_path = os.path.join(checkpoint_dir, "50.pt")
        ckpt_dict = torch.load(ckpt_path)
    else:
        ckpt_dict = checkpointer.load_checkpoint()
    read_time = round(time.time() - t0, 2)
    if "model" in ckpt_dict:
        model.load_state_dict(ckpt_dict["model"])
    if "optimizer" in ckpt_dict:
        optimizer.load_state_dict(ckpt_dict["optimizer"])
    if "sampler" in ckpt_dict:
        train_loader.sampler.load_state_dict(ckpt_dict["sampler"])
    iter_num = ckpt_dict.get("step", 0)
    load_time = round(time.time() - t0, 2)
    print(
        f"Local rank {local_rank}: reading time {read_time}, "
        f"loading time {load_time}s"
    )

    for epoch in range(args.epochs):
        # Note: set the epoch into the sampler.
        train_loader.sampler.set_epoch(epoch)
        for X, Y in train_loader:
            with elastic_trainer.step():
                # Determine and set the learning rate for this iteration
                lr = get_lr(iter_num, args) if decay_lr else learning_rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                for micro_step in range(gradient_accumulation_steps):
                    t0 = time.time()
                    X, Y = X.to(device), Y.to(device)
                    with ctx:
                        logits, loss = model(X, Y)
                        # Scale the loss to account for gradient accumulation
                        loss = loss / gradient_accumulation_steps
                    # immediately async prefetch next batch while model
                    # is doing the forward pass on the GPU
                    # Backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()
                    # Clip the gradient
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), grad_clip
                        )
                    # Step the optimizer and scaler if training in fp16
                    scaler.step(optimizer)
                    scaler.update()
                    # Flush the gradients as soon as we can,
                    # no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)

                    # Timing and logging
                    t1 = time.time()
                    dt = t1 - t0
                    total_time += dt

                    if iter_num % log_interval == 0:
                        # Get loss as float. note: this is a CPU-GPU sync point
                        # scale up to undo the division above, approximating
                        # the true total loss (exact would have been a sum)
                        lossf = loss.item() * gradient_accumulation_steps
                        if (
                            local_iter_num >= 5
                        ):  # Let the training loop settle a bit
                            mfu = raw_model.estimate_mfu(
                                batch_size * gradient_accumulation_steps, dt
                            )
                            running_mfu = (
                                mfu
                                if running_mfu == -1.0
                                else 0.9 * running_mfu + 0.1 * mfu
                            )
                        cuda_mem = torch.cuda.max_memory_allocated() / 1e9
                        log_rank0(
                            f"iter {iter_num}: loss {lossf:.4f},"
                            f" time {dt * 1000:.2f}ms, "
                            f"mfu {running_mfu * 100:.2f}%,"
                            f" cuda memory {cuda_mem:.3f}G, "
                            f"lr {lr:.2e}, total time {total_time:.2f}s"
                        )
                    iter_num += 1
                    local_iter_num += 1
                    start_save_t = time.time()
                    if args.use_native_ckpt:
                        saved = native_save_checkpoint(
                            iter_num,
                            model,
                            optimizer,
                            train_loader,
                            args.save_storage_interval,
                        )
                    else:
                        saved = flash_save_checkpoint(
                            checkpointer,
                            iter_num,
                            model,
                            optimizer,
                            train_loader,
                            args.save_memory_interval,
                            args.save_storage_interval,
                        )
                    if saved:
                        save_time = round(time.time() - start_save_t, 2)
                        print(f"Save checkpoint time {save_time}s")

                    # Termination conditions
                    if iter_num > max_iters:
                        break
                if iter_num > max_iters:
                    break
        if iter_num > max_iters:
            break


def native_save_checkpoint(
    iter_num, model, optimizer, train_loader, save_storage_interval
):
    saved = False
    if iter_num % save_storage_interval != 0:
        return saved
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": iter_num,
    }
    if isinstance(train_loader.sampler, ElasticDistributedSampler):
        sampler_sd = train_loader.sampler.state_dict(
            iter_num, train_loader.batch_size
        )
        state_dict["ds_sampler"] = sampler_sd
    ckpt_path = os.path.join(checkpoint_dir, f"{iter_num}.pt")
    torch.save(state_dict, ckpt_path)
    saved = True


def flash_save_checkpoint(
    checkpointer,
    iter_num,
    model,
    optimizer,
    train_loader,
    save_memory_interval,
    save_storage_interval,
):
    saved = False
    if (
        iter_num % save_memory_interval != 0
        and iter_num % save_storage_interval != 0
    ):
        return saved
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": iter_num,
    }
    if isinstance(train_loader.sampler, ElasticDistributedSampler):
        sampler_sd = train_loader.sampler.state_dict(
            iter_num, train_loader.batch_size
        )
        state_dict["ds_sampler"] = sampler_sd
    if iter_num % save_memory_interval == 0:
        checkpointer.save_checkpoint(
            iter_num, state_dict, storage_type=StorageType.MEMORY
        )
        saved = True
    if iter_num % save_storage_interval == 0:
        checkpointer.save_checkpoint(
            iter_num, state_dict, storage_type=StorageType.DISK
        )
        saved = True
    return saved


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    add_train_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train()
    cleanup()
