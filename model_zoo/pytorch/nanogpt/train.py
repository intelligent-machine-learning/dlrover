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
import math
import os
import pickle
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from model import GPT, GPTConfig
from torch.nn.parallel import DistributedDataParallel as DDP

local_rank = None
master_process = False


def get_data_loader(data_dir):
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(
        os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
    )
    # Attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    return train_data, val_data, meta_vocab_size


def get_batch(
    split,
    train_data,
    val_data,
    device_type="cpu",
    device="cpu",
    batch_size=12,
    block_size=128,
):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(
                (data[i : i + block_size]).astype(np.int64)  # noqa: E203 E501
            )
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (data[i + 1 : i + 1 + block_size]).astype(np.int64)  # noqa
            )
            for i in ix
        ]
    )
    if device_type == "cuda":
        # Pin arrays x,y, which allows us to move them
        # to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def gpt_init(meta_vocab_size=None, args=None):
    n_layer = args.n_layer
    n_head = args.n_head
    n_embd = args.n_embd
    block_size = args.block_size
    bias = args.bias
    dropout = args.dropout
    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )  # Start with model_args from command line
    # Init a new model from scratch
    print("Initializing a new model from scratch")
    # Determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 "
            "(50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = (
        meta_vocab_size if meta_vocab_size is not None else 50304
    )
    gptconf = GPTConfig(**model_args)
    return GPT(gptconf)


# Learning rate decay scheduler (cosine with warmup)
def get_lr(it, args):
    learning_rate = args.learning_rate
    warmup_iters = args.warmup_iters
    lr_decay_iters = args.lr_decay_iters
    min_lr = args.min_lr
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def log_rank0(msg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(msg)


def setup(args):
    global local_rank, master_process

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if use_cuda:
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"rank {rank} is initialized local_rank = {local_rank}")
    # This process will do logging, checkpointing etc.
    master_process = rank == 0
    seed_offset = rank  # Each process gets a different seed
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn


def cleanup():
    dist.destroy_process_group()


def train():
    global local_rank

    args = arg_parser()
    setup(args)
    world_size = int(os.environ["WORLD_SIZE"])
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size = args.batch_size
    block_size = args.block_size
    assert gradient_accumulation_steps % world_size == 0
    gradient_accumulation_steps //= world_size
    tokens_per_iter = (
        gradient_accumulation_steps * world_size * batch_size * block_size
    )  # noqa: E501
    log_rank0(f"tokens per iteration will be: {tokens_per_iter:,}")
    # data
    train_data, val_data, meta_vocab_size = get_data_loader(args.data_dir)
    device = (
        f"cuda:{local_rank}"
        if torch.cuda.is_available() and "cuda" in args.device
        else "cpu"
    )
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
    model = gpt_init(meta_vocab_size, args=args)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
    model = model.to(device)
    # Optimizer
    print(f"creating optimizer...{model.parameters()}")
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    if torch.cuda.is_available() and device_type == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Running basic DDP example on local rank {local_rank}.")
        # Create model and move it to GPU with id rank
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
        print(f"Model device {model.device}")
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Running basic CPU example on device {device}.")
        model = model.to(device)
        model = DDP(model)
        print(f"Model device {model.device}")

    # Compile the model
    if compile == "True":
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    # Training loop
    X, Y = get_batch(
        "train", train_data, val_data, device_type
    )  # Fetch the very first batch
    total_time = 0.0
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    raw_model = model.module  # Unwrap DDP container if needed
    running_mfu = -1.0
    iter_num = 0
    decay_lr = args.decay_lr
    max_iters = args.max_iters
    log_interval = args.log_interval
    grad_clip = args.grad_clip
    learning_rate = args.learning_rate

    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t0 = time.time()
        # Forward backward update, with optional gradient accumulation
        # to simulate larger batch size and using the GradScaler
        # if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = (
                    loss / gradient_accumulation_steps
                )  # Scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model
            # is doing the forward pass on the GPU
            X, Y = get_batch("train", train_data, val_data, device_type)
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
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
                f"mfu {running_mfu*100:.2f}%, lr {lr:.2e}, "
                f"total time {total_time:.2f}s"
            )
        iter_num += 1
        local_iter_num += 1

        # Termination conditions
        if iter_num > max_iters:
            break


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")

    # Data settings
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out", required=False)
    parser.add_argument(
        "--eval_interval", type=int, default=2000, required=False
    )
    parser.add_argument("--log_interval", type=int, default=1, required=False)
    parser.add_argument("--eval_iters", type=int, default=200, required=False)
    parser.add_argument("--eval_only", action="store_true", required=False)
    parser.add_argument(
        "--always_save_checkpoint", action="store_true", required=False
    )
    parser.add_argument("--batch_size", type=int, default=12, required=False)
    parser.add_argument("--block_size", type=int, default=1024, required=False)

    # Model settings
    parser.add_argument("--n_layer", type=int, default=6, required=False)
    parser.add_argument("--n_head", type=int, default=6, required=False)
    parser.add_argument("--n_embd", type=int, default=384, required=False)
    parser.add_argument("--dropout", type=float, default=0.0, required=False)
    parser.add_argument("--bias", action="store_true", required=False)

    # Optimizer settings
    parser.add_argument(
        "--learning_rate", type=float, default=6e-4, required=False
    )
    parser.add_argument("--max_iters", type=int, default=10, required=False)
    parser.add_argument(
        "--weight_decay", type=float, default=1e-1, required=False
    )
    parser.add_argument("--beta1", type=float, default=0.9, required=False)
    parser.add_argument("--beta2", type=float, default=0.95, required=False)
    parser.add_argument("--grad_clip", type=float, default=1.0, required=False)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, required=False
    )

    # Learning rate decay settings
    parser.add_argument("--decay_lr", action="store_true", required=False)
    parser.add_argument("--warmup_iters", type=int, default=0, required=False)
    parser.add_argument(
        "--lr_decay_iters", type=int, default=10, required=False
    )
    parser.add_argument("--min_lr", type=float, default=6e-5, required=False)

    # System settings
    parser.add_argument("--device", type=str, default="cpu", required=False)
    parser.add_argument("--compile", type=str, default="False", required=False)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    train()
    cleanup()
