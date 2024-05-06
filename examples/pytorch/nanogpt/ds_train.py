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

dlrover-run --max_restarts=2 --nproc_per_node=2 \
    ds_train.py --n_layer 36 --n_head 20 --n_embd 1280 \
    --data_dir './' --ds_config ./ds_config.json \
    --epochs 50 --save_memory_interval 50 --save_storage_interval 500
"""

import argparse
import contextlib
import os
import time

import deepspeed
import torch
from lora import apply_lora
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

from dlrover.trainer.torch.flash_checkpoint.deepspeed import (
    DeepSpeedCheckpointer,
    StorageType,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def train():
    args = arg_parser()
    checkpoint_dir = args.save_dir
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
    # Optimizer
    log_rank0(f"creating optimizer...{model.parameters()}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )

    lora_config = create_lora_config(args)
    if lora_config is not None:
        log_rank0(f"apply lora config {lora_config}")
        apply_lora(model, **lora_config)
    if torch.cuda.is_available() and device_type == "cuda":
        # Create model and move it to GPU with id rank
        model = model.to(local_rank)
        model, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            config=args.ds_config,
        )
    else:
        raise ValueError("Deepspeed must run with cuda devices.")

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
    learning_rate = args.learning_rate

    # Forward backward update, with optional gradient accumulation
    # to simulate larger batch size and using the GradScaler
    # if data type is float16
    t0 = time.time()
    if args.use_native_ckpt:
        model.load_checkpoint(checkpoint_dir)
    else:
        checkpointer = DeepSpeedCheckpointer(model, checkpoint_dir)
        checkpointer.load_checkpoint(checkpoint_dir)
    load_time = round(time.time() - t0, 2)
    print(f"Load checkpoint time {load_time}s")
    iter_num = model.global_steps
    print(f"The restored iteration step is {iter_num}")

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
            model.backward(loss)
            model.step()

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
            start_save_t = time.time()
            if args.use_native_ckpt:
                saved = native_save_checkpoint(
                    model, checkpoint_dir, iter_num, args.save_storage_interval
                )
            else:
                saved = flash_save_checkpoint(
                    checkpointer,
                    iter_num,
                    args.save_memory_interval,
                    args.save_storage_interval,
                    checkpoint_dir,
                )
            if saved:
                save_time = round(time.time() - start_save_t, 2)
                print(f"Save checkpoint time {save_time}s")

            # Termination conditions
            if iter_num > max_iters:
                break

        if iter_num > max_iters:
            break


def native_save_checkpoint(
    model: deepspeed.DeepSpeedEngine,
    checkpoint_dir,
    iter_num,
    save_storage_interval,
):
    saved = False
    if iter_num % save_storage_interval == 0:
        model.save_checkpoint(checkpoint_dir, tag=iter_num)
        saved = True
    return saved


def flash_save_checkpoint(
    checkpointer: DeepSpeedCheckpointer,
    iter_num,
    save_memory_interval,
    save_storage_interval,
    checkpoint_dir,
):
    saved = False
    if iter_num % save_memory_interval == 0:
        saved = True
        checkpointer.save_checkpoint(
            checkpoint_dir,
            tag=iter_num,
            storage_type=StorageType.MEMORY,
        )
    if iter_num % save_storage_interval == 0:
        saved = True
        checkpointer.save_checkpoint(
            checkpoint_dir, tag=iter_num, storage_type=StorageType.DISK
        )
    return saved


# Determine the device type based on the input string.
def device_type(string):
    lower_string = string.lower()
    if "gpu" in lower_string or "cuda" in lower_string:
        if lower_string != "cuda":
            log_rank0(
                "It seems you are trying to use a cuda device."
                'The correct argument should be "cuda".'
                "Automatically using the cuda device."
            )
        return "cuda"
    else:
        if lower_string != "cpu":
            log_rank0(
                f'Unrecognized device type argument "{lower_string}".'
                "Defaulting to use the cpu device."
            )
        return "cpu"


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")

    add_train_args(parser)
    parser.add_argument("--ds_config", type=str, default="", required=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    train()
    cleanup()
