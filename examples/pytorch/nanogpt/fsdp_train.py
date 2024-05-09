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
The start command on a local node:

dlrover-run --nproc_per_node=2 fsdp_train.py \
    --n_layer 48 --n_head 16 --n_embd 384 --data_dir './result' \
    --epochs 50 --save_memory_interval 50 --save_storage_interval 500
"""

import argparse
import contextlib
import functools
import os
import time

import torch
import torch.distributed.checkpoint as dist_ckpt
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
    FsdpFullCheckpointer,
    FsdpShardCheckpointer,
    StorageType,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def train(args, train_params):
    """
    Train the model with the given parameters that has been set up correctly.

    Args:
        args: Arguments parsed from the command line.
        train_params:  The parameters set up for training.
            - env_params:       Parameters relating to the environment.
            - model_params:     Parameters relating to the model.
            - ckpt_params:      Parameters relating to the checkpoint.
    """
    (env_params, model_params, ckpt_params) = train_params
    # Load from checkpoint.
    load_checkpoint(model_params, ckpt_params)

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

        # Update total_steps.
        nonlocal total_steps
        total_steps += 1
        model_params["total_steps"] = total_steps

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
                optimizer.zero_grad(set_to_none=True)
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
            # Step with gradient accumulation.
            step_grad_accum(idx, data, target)
            # Save the checkpoint. Update the total steps.
            save_checkpoint(model_params, ckpt_params)

            # Termination conditions
            if total_steps > args.max_iters:
                return


def setup_train_params(args) -> tuple:
    """
    Set up all the necessary parameters before training.

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

    # Set up the model.
    if "cuda" in device:
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

    # Set up the optimizer.
    log_rank0(f"Creating optimizer... {model.parameters()}")
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
    )

    # Compile the model.
    if compile == "True":
        log_rank0("Compiling the model... (takes a ~minute).")
        model = torch.compile(model)  # requires PyTorch 2.0

    # Set up the ElasticTrainer.
    elastic_trainer = ElasticTrainer(
        model=model,
        dataloader=train_loader,
    )
    optimizer = elastic_trainer.prepare(optimizer)

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
        "total_steps": 0,
        "train_loader": train_loader,
        "elastic_trainer": elastic_trainer,
        "optimizer": optimizer,
    }

    ckpt_params = {
        "use_native": args.use_native_ckpt,
        "flash_full_ckpt": args.flash_full_ckpt,
        "checkpoint_dir": args.save_dir,
        "checkpointer": None,
        "save_memory_interval": args.save_memory_interval,
        "save_storage_interval": args.save_storage_interval,
    }

    # Return as a tuple of dictionaries.
    return (env_params, model_params, ckpt_params)


def timing_logger(func):
    """
    Decorator to time and log the function execution.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        total_time = time.time() - start_time

        if func == load_checkpoint:
            # Print the load checkpoint time.
            with result as loaded:
                print(
                    f"Load checkpoint time : {total_time}s"
                ) if loaded else None
        elif func == save_checkpoint:
            # Print the save checkpoint time.
            with result as saved:
                print(
                    f"Save checkpoint time: {total_time}s"
                ) if saved else None

        return result

    return wrapper


@timing_logger
def load_checkpoint(model_params, ckpt_params):
    """
    Load the checkpoint to memory or disk when needed.

    Returns: A boolean value indicating whether the checkpoint was loaded.
            This result is mainly used by the "timer" decorator.
    """
    loaded = False
    model = model_params["model"]
    optimizer = model_params["optimizer"]
    checkpoint_dir = ckpt_params["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    if ckpt_params["use_native"]:
        # If using native checkpointing.
        path = os.path.join(checkpoint_dir, str(model_params["total_steps"]))
        if os.path.exists(path):
            # Load model state dict.
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                state_dict = {
                    "model": model.state_dict(),
                    "step": 0,
                    # Cannot load the optimizer state_dict
                    # together with the model state_dict.
                }
            storage_reader = dist_ckpt.FileSystemReader(path)
            dist_ckpt.load_state_dict(
                state_dict=state_dict,
                storage_reader=storage_reader,
            )
            model.load_state_dict(state_dict["model"])

            # Load optimizer state dict.
            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=storage_reader,
            )
            flattened_osd = FSDP.optim_state_dict_to_load(
                model, optimizer, optim_state["optim"]
            )
            optimizer.load_state_dict(flattened_osd)

            # Update model params.
            model_params["model"] = model
            model_params["optimizer"] = optimizer
            model_params["total_steps"] = state_dict["step"]
            loaded = True

    else:
        # If using flash checkpointing.
        if ckpt_params["flash_full_ckpt"]:
            checkpointer = FsdpFullCheckpointer(checkpoint_dir)
        else:
            checkpointer = FsdpShardCheckpointer(checkpoint_dir)
        extra_sd = checkpointer.load_checkpoint(model, optimizer)

        # Update model params.
        model_params["total_steps"] = extra_sd.get("step", 0)
        ckpt_params["checkpointer"] = checkpointer
        loaded = True

    return loaded


@timing_logger
def save_checkpoint(model_params, ckpt_params):
    """
    Save the checkpoint to memory or disk when needed.

    Returns: A boolean value indicating whether the checkpoint was saved.
            This result is mainly used by the "timer" decorator.
    """
    saved = False
    model = model_params["model"]
    steps = model_params["total_steps"]
    optimizer = model_params["optimizer"]
    checkpointer = ckpt_params["checkpointer"]
    checkpoint_dir = ckpt_params["checkpoint_dir"]

    # Save the checkpoint.
    if ckpt_params["use_native"]:
        # If using native checkpointing.
        if steps % ckpt_params["save_storage_interval"] == 0:
            # Get state dict.
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                state_dict = {
                    "model": model.state_dict(),
                    "optim": FSDP.optim_state_dict(model, optimizer),
                    "step": steps,
                }

            # Save state dict.
            path = os.path.join(checkpoint_dir, str(steps))
            dist_ckpt.save_state_dict(
                state_dict=state_dict,
                storage_writer=dist_ckpt.FileSystemWriter(path=path),
            )
        saved = True

    else:
        # If using flash checkpointing.
        # Warning: When n_procs_per_node is not greater than 1,
        # the checkpoint saving would be stuck.
        extra_sd = {"step": steps}
        if steps % ckpt_params["save_memory_interval"] == 0:
            checkpointer.save_checkpoint(
                steps,
                model,
                optimizer,
                extra_sd,
                storage_type=StorageType.MEMORY,
            )
            saved = True

        if steps % ckpt_params["save_storage_interval"] == 0:
            checkpointer.save_checkpoint(
                steps,
                model,
                optimizer,
                extra_sd,
                storage_type=StorageType.DISK,
            )
            saved = True

    return saved


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    add_train_args(parser)

    parser.add_argument("--cpu_offload", action="store_true", required=False)
    parser.add_argument(
        "--flash_full_ckpt", action="store_true", required=False
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    train_params = setup_train_params(args)
    train(args, train_params)
    cleanup()
