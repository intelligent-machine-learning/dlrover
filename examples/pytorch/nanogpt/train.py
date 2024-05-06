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

dtypes = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def train(args, train_params):
    """
    Train the model with the given parameters that has been set up correctly.

    Args:
        args:  Arguments parsed from the command line.
        train_params:  dtype_name, grad_accum_steps, total_steps, device, train_loader,
                        elastic_trainer, model, optimizer, checkpointer
    """
    (dtype_name, grad_accum_steps, total_steps, device, train_loader, elastic_trainer,
     model, optimizer, checkpointer) = train_params

    total_time = 0.0
    local_iter_num = 0  # Number of iterations in the lifetime of this process
    running_mfu = -1.0

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype_name == ("float16" or "bfloat16")))

    for epoch in range(args.epochs):
        # Set learning rate.
        learning_rate = get_lr(total_steps, args) if args.decay_lr else args.learning_rate

        # Set epoch into the sampler.
        train_loader.sampler.set_epoch(epoch)
        # Set learning rate.
        optimizer.param_groups[0]["lr"] = learning_rate

        for _, (data, target) in enumerate(train_loader):
            with elastic_trainer.step():
                start_time = time.time()
                for micro_step in range(grad_accum_steps):
                    data, target = data.to(device), target.to(device)

                    # Set context.
                    if "cpu" in device:
                        context = contextlib.nullcontext()
                    else:
                        context = torch.amp.autocast(device.split(':')[0], dtypes[dtype_name])

                    with context:
                        _, loss = model(data, target)
                        # Scale the loss to account for gradient accumulation
                        loss = loss / grad_accum_steps
                    # immediately async prefetch next batch while model
                    # is doing the forward pass on the GPU
                    # Backward pass, with gradient scaling if training in fp16
                    scaler.scale(loss).backward()

                    # Clip the gradient
                    if args.grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                    # Step the optimizer and scaler if training in fp16
                    scaler.step(optimizer)
                    scaler.update()
                    # Flush the gradients as soon as we can,
                    # no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)

                    # Timing and logging
                    dt = time.time() - start_time
                    total_time += dt

                    if total_steps % args.log_interval == 0:
                        # Get loss as float. note: this is a CPU-GPU sync point
                        # scale up to undo the division above, approximating
                        # the true total loss (exact would have been a sum)
                        lossf = loss.item() * grad_accum_steps
                        if local_iter_num >= 5:
                            # Let the training loop settle a bit
                            mfu = model.module.estimate_mfu(
                                args.batch_size * grad_accum_steps, dt
                            )

                            if running_mfu == -1.0:
                                running_mfu = mfu
                            else:
                                running_mfu = 0.9 * running_mfu + 0.1 * mfu

                        cuda_mem = torch.cuda.max_memory_allocated() / 1e9
                        log_rank0(
                            f"iter {total_steps}: loss {lossf:.4f}, "
                            f"time {dt * 1000:.2f}ms, "
                            f"mfu {running_mfu * 100:.2f}%, "
                            f"cuda memory {cuda_mem:.3f}G, "
                            f"lr {learning_rate:.2e}, total time {total_time:.2f}s"
                        )
                    total_steps += 1
                    local_iter_num += 1

                    # Save the checkpoint.
                    start_save_t = time.time()
                    if args.use_native_ckpt:
                        saved = native_save_checkpoint(
                            total_steps,
                            model,
                            optimizer,
                            train_loader,
                            args.save_storage_interval,
                            args.checkpoint_dir,
                        )
                    else:
                        saved = flash_save_checkpoint(
                            checkpointer,
                            total_steps,
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
                    if total_steps > args.max_iters:
                        break
                if total_steps > args.max_iters:
                    break
        if total_steps > args.max_iters:
            break


def setup_everything(args) -> tuple:
    """
    Set up all the necessary components before training.

    Returns:
        tuple: A tuple containing...
                dtype_name:  The name of the data type.
                grad_accum_steps:  The number of gradient accumulation steps.
                total_steps:  The total number of steps.
                device:  The device to train on.
                train_loader:  The data loader for training.
                elastic_trainer:  The ElasticTrainer object.
                model:  The model to train.
                optimizer:  The optimizer to use.
                checkpointer:  The checkpointer object.
    """
    setup()
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    dtype_name = "float16"
    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        if torch.cuda.is_bf16_supported():
            dtype_name = "bfloat16"
    else:
        device = "cpu"

    # Set up the gradient accumulation steps.
    total_steps = args.gradient_accumulation_steps
    if total_steps == 0:
        grad_accum_steps = 1
    else:
        grad_accum_steps = total_steps // world_size

    # tokens_per_iter = grad_accum_steps * world_size * args.batch_size * args.block_size
    # log_rank0(f"tokens per iteration will be: {tokens_per_iter:,}")

    train_loader, _, vocab_size = get_data_loaders(args.data_dir, args.batch_size, args.block_size)
    model = gpt_init(vocab_size, args=args)
    model = model.to(device)

    # Apply LoRA if needed.
    lora_config = create_lora_config(args)
    if lora_config is not None:
        log_rank0(f"apply lora config {lora_config}")
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
        log_rank0("Compiling the model... (takes a ~minute)")
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
    model.load_state_dict(ckpt_dict["model"]) if "model" in ckpt_dict else None
    optimizer.load_state_dict(ckpt_dict["optimizer"]) if "optimizer" in ckpt_dict else None
    train_loader.sampler.load_state_dict(ckpt_dict["sampler"]) if "sampler" in ckpt_dict else None
    total_steps = ckpt_dict.get("step", 0)
    # Print the checkpointer loading time.
    print(f"Local rank {local_rank}: checkpointer loading time {round(time.time() - t0, 2)}s")

    return (dtype_name, grad_accum_steps, total_steps, device, train_loader, elastic_trainer,
            model, optimizer, checkpointer)


def native_save_checkpoint(
        total_steps,
        model,
        optimizer,
        train_loader,
        save_storage_interval,
        checkpoint_dir,
):
    if total_steps % save_storage_interval != 0:
        return False

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": total_steps,
    }
    if isinstance(train_loader.sampler, ElasticDistributedSampler):
        sampler_sd = train_loader.sampler.state_dict(total_steps, train_loader.batch_size)
        state_dict["ds_sampler"] = sampler_sd

    ckpt_path = os.path.join(checkpoint_dir, f"{total_steps}.pt")
    torch.save(state_dict, ckpt_path)
    return True


def flash_save_checkpoint(
        checkpointer,
        total_steps,
        model,
        optimizer,
        train_loader,
        save_memory_interval,
        save_storage_interval,
):
    if (total_steps % save_memory_interval != 0) and (total_steps % save_storage_interval != 0):
        return False

    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": total_steps,
    }
    if isinstance(train_loader.sampler, ElasticDistributedSampler):
        sampler_sd = train_loader.sampler.state_dict(total_steps, train_loader.batch_size)
        state_dict["ds_sampler"] = sampler_sd

    # Save the checkpoint to memory or disk.
    if total_steps % save_memory_interval == 0:
        checkpointer.save_checkpoint(total_steps, state_dict, storage_type=StorageType.MEMORY)
    if total_steps % save_storage_interval == 0:
        checkpointer.save_checkpoint(total_steps, state_dict, storage_type=StorageType.DISK)
    return True


def arg_parser():
    parser = argparse.ArgumentParser(description="Process training parameters")
    add_train_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    train_params = setup_everything(args)
    train(args, train_params)
    cleanup()
