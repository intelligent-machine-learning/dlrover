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
    --n_layer 48 --n_head 16 --n_embd 1600 --data_dir "./" \
    --epochs 50 --save_memory_interval 50 --save_storage_interval 500
"""

import argparse
import contextlib
import functools
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


def train(args, train_params):
    """
    Train the model with the given parameters that has been set up correctly.

    Args:
        args:  Arguments parsed from the command line.
        train_params:  The parameters set up for training.
            - env_params:       Parameters relating to the environment.
            - model_params:     Parameters relating to the model.
            - ckpt_params:      Parameters relating to the checkpoint.
    """
    (env_params, model_params, ckpt_params) = train_params
    # Unpack the parameters for model training.
    model = model_params["model"]
    context = model_params["context"]
    scaler = model_params["scaler"]
    device = model_params["device"]
    grad_accum_steps = model_params["grad_accum_steps"]
    total_steps = model_params["total_steps"]  # The only mutable variable.
    train_loader = model_params["train_loader"]
    elastic_trainer = model_params["elastic_trainer"]
    optimizer = model_params["optimizer"]

    previous_mfu = -1.0
    total_time = 0.0
    run_time = 0.0

    def grad_accum_logger(step_with_grad_accum):
        """
        An inner function working as a decorator to log the training process.
        """

        @functools.wraps(step_with_grad_accum)
        def wrapper(idx, data, target):
            nonlocal previous_mfu, run_time, total_time

            start_time = time.time()
            print_log, loss = step_with_grad_accum(idx, data, target)
            run_time += time.time() - start_time
            total_time += run_time

            if print_log:
                # Estimate the model flops utilization (MFU).
                mfu = model.module.estimate_mfu(
                    args.batch_size * grad_accum_steps, run_time
                )
                if idx > 5:
                    if previous_mfu == -1.0:
                        previous_mfu = mfu
                    else:
                        previous_mfu = 0.9 * previous_mfu + 0.1 * mfu

                # Estimate the CUDA memory usage.
                cuda_mem = torch.cuda.memory_allocated() / 1e9

                # Print log.
                print(
                    f"iter {total_steps}: loss {loss:.4f}, "
                    f"time {run_time * 1000:.2f}ms, "
                    f"mfu {previous_mfu * 100:.2f}%, "
                    f"cuda memory {cuda_mem:.3f}G, "
                    f"lr {learning_rate:.2e}, "
                    f"total time {total_time:.2f}s"
                )
                run_time = 0
            return print_log, loss

        return wrapper

    @grad_accum_logger
    def step_grad_accum(idx, data, target):
        """
        An inner function to perform training with gradient accumulation.
        """
        print_log = False
        data, target = data.to(device), target.to(device)

        with elastic_trainer.step():
            # Forward pass.
            with context:
                _, loss = model(data, target)
            # Scale the loss for gradient accumulation.
            loss = loss / grad_accum_steps
            # Backward pass, with gradient scaling.
            scaler.scale(loss).backward()
            # Clip gradients.
            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            # Weight update
            if (idx + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                print_log = True
        return print_log, loss.item() * grad_accum_steps

    # Training loop.
    for epoch in range(args.epochs):
        # Set epoch into the sampler.
        train_loader.sampler.set_epoch(epoch)
        # Set learning rate.
        learning_rate = (
            get_lr(total_steps, args) if args.decay_lr else args.learning_rate
        )
        optimizer.param_groups[0]["lr"] = learning_rate

        # Training loop.
        for idx, (data, target) in enumerate(train_loader):
            # Update total_steps.
            total_steps += 1
            model_params["total_steps"] = total_steps
            # Step with gradient accumulation.
            step_grad_accum(idx, data, target)
            # Save the checkpoint. Update the total steps.
            save_checkpoint(model_params, ckpt_params)

            # Termination conditions
            if total_steps > args.max_iters:
                return


def setup_train_params(args) -> tuple:
    """
    Set up all the necessary components before training.

    Returns:
        tuple: A tuple containing three dictionaries:
            - env_params:       Parameters relating to the environment.
            - model_params:     Parameters relating to the model.
            - ckpt_params:      Parameters relating to the checkpoint.
    """
    setup()
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }

    dtype_name = "float16"
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        if torch.cuda.is_bf16_supported():
            dtype_name = "bfloat16"
        context = torch.amp.autocast("cuda", dtypes[dtype_name])
    else:
        device = "cpu"
        context = contextlib.nullcontext()
    scaler = torch.cuda.amp.GradScaler(
        enabled=(dtype_name == ("float16" or "bfloat16"))
    )

    # Set up the gradient accumulation steps.
    grad_accum_steps = args.gradient_accumulation_steps
    if (grad_accum_steps == 0) or (grad_accum_steps // world_size == 0):
        grad_accum_steps = 1
    else:
        grad_accum_steps = grad_accum_steps // world_size

    tokens_per_iter = (
        grad_accum_steps * world_size * args.batch_size * args.block_size
    )
    log_rank0(f"Tokens per iteration will be: {tokens_per_iter:,}")

    train_loader, _, vocab_size = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )
    model = gpt_init(vocab_size, args=args)
    model = model.to(device)

    # Apply LoRA if needed.
    lora_config = create_lora_config(args)
    if lora_config is not None:
        log_rank0(f"Apply lora config {lora_config}.")
        apply_lora(model, **lora_config)

    # Set up the model.
    if torch.cuda.is_available():
        print(f"Running basic DDP example on {device}.")
        model = DDP(model, device_ids=[local_rank])
        print(f"Model device {model.device}")
    else:
        print(f"Running basic CPU example on {device}.")
        model = DDP(model)
        print(f"Model device {model.device}")

    if compile == "True":
        log_rank0("Compiling the model... (takes a ~minute).")
        model = torch.compile(model)  # requires PyTorch 2.0

    # Set up the ElasticTrainer.
    elastic_trainer = ElasticTrainer(
        model=model,
        dataloader=train_loader,
    )

    # Set up the optimizer.
    log_rank0(f"Creating optimizer... {model.parameters()}")
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )
    optimizer = elastic_trainer.prepare(optimizer)

    # Load from checkpointer.
    t0 = time.time()
    checkpoint_dir = args.save_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpointer = DdpCheckpointer(checkpoint_dir)

    if args.use_native_ckpt:
        ckpt_dict = torch.load(os.path.join(checkpoint_dir, "50.pt"))
    else:
        ckpt_dict = checkpointer.load_checkpoint()
    # Load {model}, {optimizer}, {sampler} and {step} from the checkpoint.
    if "model" in ckpt_dict:
        model.load_state_dict(ckpt_dict["model"])
    if "optimizer" in ckpt_dict:
        optimizer.load_state_dict(ckpt_dict["optimizer"])
    if "sampler" in ckpt_dict:
        train_loader.sampler.load_state_dict(ckpt_dict["sampler"])
    total_steps = ckpt_dict.get("step", 0)
    # Print the checkpointer loading time.
    print(
        f"Local rank {local_rank}: "
        f"checkpointer loading time {round(time.time() - t0, 2)}s"
    )

    # Prepare the parameters for training.
    env_params = {
        "world_size": world_size,
        "local_rank": local_rank,
    }

    model_params = {
        "model": model,
        "context": context,
        "scaler": scaler,
        "device": device,
        "grad_accum_steps": grad_accum_steps,
        "total_steps": total_steps,
        "train_loader": train_loader,
        "elastic_trainer": elastic_trainer,
        "optimizer": optimizer,
    }

    ckpt_params = {
        "use_native": args.use_native_ckpt,
        "checkpointer": checkpointer,
        "ckpt_dir": checkpoint_dir,
        "save_memory_interval": args.save_memory_interval,
        "save_storage_interval": args.save_storage_interval,
    }

    # Return as a tuple of dictionaries.
    return (env_params, model_params, ckpt_params)


def timer(func):
    """
    Decorator to time the function execution.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        total_time = time.time() - start_time

        if func == save_checkpoint:
            # Print the save checkpoint time.
            with result as saved:
                print(f"Save checkpoint time: {total_time}") if saved else None
        return result

    return wrapper


@timer
def save_checkpoint(model_params, ckpt_params):
    """
    Save the checkpoint to memory or disk when needed.

    Returns: A boolean value indicating whether the checkpoint was saved.
            This result is mainly used by the "timer" decorator.
    """
    saved = False
    steps = model_params["total_steps"]

    def prepare_state_dict():
        """
        An inner function to prepare the state dictionary for saving.
        """
        state_dict = {
            "model": model_params["model"].state_dict(),
            "optimizer": model_params["optimizer"].state_dict(),
            "step": steps,
        }

        train_loader = model_params["train_loader"]
        if isinstance(train_loader.sampler, ElasticDistributedSampler):
            sampler_sd = train_loader.sampler.state_dict(
                steps, train_loader.batch_size
            )
            state_dict["ds_sampler"] = sampler_sd

        return state_dict

    # Save the checkpoint.
    if ckpt_params["use_native"]:
        # If using native checkpointing, save the checkpoint to disk.
        if steps % ckpt_params["save_memory_interval"] == 0:
            state_dict = prepare_state_dict()
            ckpt_path = os.path.join(
                ckpt_params["ckpt_dir"], f"{model_params['total_steps']}.pt"
            )
            torch.save(state_dict, ckpt_path)
            saved = True
    else:
        # If using Flash Checkpointing, save the checkpoint to memory or disk.
        if steps % ckpt_params["save_memory_interval"] == 0:
            state_dict = prepare_state_dict()
            checkpointer = ckpt_params["checkpointer"]
            checkpointer.save_checkpoint(
                steps, state_dict, storage_type=StorageType.MEMORY
            )
            saved = True

        if steps % ckpt_params["save_storage_interval"] == 0:
            state_dict = prepare_state_dict()
            checkpointer = ckpt_params["checkpointer"]
            checkpointer.save_checkpoint(
                steps, state_dict, storage_type=StorageType.DISK
            )
            saved = True

    return saved


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    add_train_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    train_params = setup_train_params(args)
    train(args, train_params)
    cleanup()
