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


import argparse
import contextlib
import functools
import os
import time

import torch
import torch.distributed.checkpoint as dist_cp
from model import Block
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from train_utils import (
    add_train_args,
    cleanup,
    get_data_loaders,
    get_lr,
    gpt_init,
    log_rank0,
    setup,
)

from dlrover.trainer.torch.elastic.trainer import ElasticTrainer
from dlrover.trainer.torch.flash_checkpoint.fsdp import (
    FsdpCheckpointer,
    StorageType,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# We should use a shared storage to persist the checkpiont.
checkpoint_dir = "/nas/nanogpt-ckpt-fsdp/"


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
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # For later use in torch.autocast
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
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    if torch.cuda.is_available() and device_type == "cuda":
        print(f"Running basic FSDP example on local rank {local_rank}.")
        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block},
        )
        cpu_offload = (
            CPUOffload(offload_params=True) if args.cpu_offload else None
        )
        model = FSDP(
            model,
            device_id=local_rank,
            auto_wrap_policy=my_auto_wrap_policy,
            cpu_offload=cpu_offload,
        )

    else:
        raise ValueError("FSDP can only runs on CUDA.")

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

    checkpointer = FsdpCheckpointer(checkpoint_dir)

    iter_num = load_checkpoint(checkpointer, model, optimizer)
    iter_num = 0 if not iter_num else iter_num

    for epoch in range(args.epochs):
        # Note: set the epoch into the sampler.
        train_loader.sampler.set_epoch(epoch)
        for X, Y in train_loader:
            # Determine and set the learning rate for this iteration
            lr = get_lr(iter_num, args) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                if local_iter_num >= 5:  # Let the training loop settle a bit
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
            save_checkpoint(
                checkpointer,
                iter_num,
                model,
                optimizer,
                args.save_memory_interval,
                args.save_storage_interval,
            )
            # Termination conditions
            if iter_num > max_iters:
                break
        if iter_num > max_iters:
            break
        if iter_num > max_iters:
            break


def load_checkpoint(checkpointer, model, optimizer):
    # Load model_2 with parameters saved in CHECKPOINT_DIR
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
            "step": 0,
            # cannot load the optimizer state_dict together
            # with the model state_dict
        }
        storage_reader = checkpointer.get_storage_reader()
        if not storage_reader:
            return
        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=storage_reader,
        )
        model.load_state_dict(state_dict["model"])

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=storage_reader,
        )

        flattened_osd = FSDP.optim_state_dict_to_load(
            model, optimizer, optim_state["optim"]
        )
        optimizer.load_state_dict(flattened_osd)
        return state_dict["step"]


def save_checkpoint(
    checkpointer,
    step,
    model,
    optimizer,
    save_memory_interval,
    save_storage_interval,
):
    if step % save_memory_interval != 0 and step % save_storage_interval != 0:
        return
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
            "optim": FSDP.optim_state_dict(model, optimizer),
            "step": step,
        }
        ckpt_dir = os.path.join(checkpoint_dir, str(step))
        print(f"save checkpoint to  {ckpt_dir}")
        if step % save_memory_interval == 0:
            checkpointer.save_checkpoint(
                step, state_dict, ckpt_dir, storage_type=StorageType.MEMORY
            )
        if step % save_storage_interval == 0:
            checkpointer.save_checkpoint(
                step, state_dict, ckpt_dir, storage_type=StorageType.DISK
            )


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    add_train_args(parser)

    parser.add_argument("--cpu_offload", action="store_true", required=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    train()
    cleanup()
