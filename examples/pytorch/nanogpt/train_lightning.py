# Copyright 2023 The DLRover Authors. All rights reserved. Licensed
# under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import argparse
import contextlib
import json
import math
import os
import pickle
from datetime import timedelta

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from model import GPT, GPTConfig
from torch.utils.data import DataLoader, Dataset

from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler

local_rank = None


class UpdateDataStepCallback(pl.Callback):
    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        trainer.datamodule.global_step = trainer.global_step

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        trainer.datamodule.train_sampler.set_epoch(trainer.current_epoch)


class Nanogpt(pl.LightningModule):
    def __init__(self, args, ctx=None, dataset_vocab_size=None):
        super(Nanogpt, self).__init__()
        ptdtype = self._get_autocast_dtype()
        self.device_type = self._device_type(args.device)
        # Initialize ctx based on the device_type
        self.ctx = (
            contextlib.nullcontext()
            if self.device_type == "cpu"
            else torch.amp.autocast(
                device_type=self.device_type, dtype=ptdtype
            )
        )
        self.automatic_optimization = False
        self.meta_vocab_size = self._get_meta_vocab_size(
            ckpt_dir=args.checkpoint_dir, data_dir=args.data_dir
        )
        self.model = self._gpt_init(
            meta_vocab_size=self.meta_vocab_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            bias=args.bias,
            dropout=args.dropout,
        )

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.long()
        with self.ctx:
            logits, loss = self.model(x, y)
        # Control gradient accumulation using the
        # accumulate_grad_batches parameter
        loss = loss / self.trainer.accumulate_grad_batches
        # Manually compute gradients
        self.manual_backward(loss)
        # Check if it's time for an optimizer step
        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        if ((self.trainer.global_step + 1) % accumulate_grad_batches) == 0:
            # Perform optimizer step and zero_grad
            self.optimizers().step()
            self.optimizers().zero_grad()
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.long()
        logits, loss = self.model(x)
        # self.log("val_loss", loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            weight_decay=args.weight_decay,
            learning_rate=args.learning_rate,
            betas=(args.beta1, args.beta2),
            device_type=self.device_type,
        )
        return optimizer

    def _get_autocast_dtype(self):
        # Check if CUDA is available and bfloat16 is supported
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = "bfloat16"
        else:
            dtype = "float16"

        # Map dtype to corresponding PyTorch datatype
        ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[dtype]

        return ptdtype

    def _get_meta_vocab_size(self, ckpt_dir, data_dir):
        if ckpt_dir is not None:
            # Load the existing meta_vocab_size
            # Attempt to derive vocab_size from the dataset
            config_path = os.path.join(ckpt_dir, "config.json")
            meta_vocab_size = None
            with open(config_path) as f:
                config = json.load(f)
                meta_vocab_size = config["vocab_size"]
                print(
                    f"found vocab_size = {meta_vocab_size}"
                    f"(inside {config_path})"
                )
        else:
            # Determine the vocab size we'll use for from-scratch training
            # Attempt to derive vocab_size from the dataset
            meta_path = os.path.join(data_dir, "meta.pkl")
            meta_vocab_size = None
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        if meta_vocab_size is None:
            print(
                "defaulting to vocab_size of GPT-2 to 50304 "
                "(50257 rounded up for efficiency)"
            )
        meta_vocab_size = (
            meta_vocab_size if meta_vocab_size is not None else 50304
        )
        return meta_vocab_size

    def _gpt_init(
        self,
        meta_vocab_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        bias=True,
        dropout=0.1,
    ):
        # model init
        model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            vocab_size=meta_vocab_size,
            dropout=dropout,
        )  # Start with model_args from command line
        # Init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        return GPT(gptconf)

    # Learning rate decay scheduler (cosine with warmup)
    def _get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
        # 1) Linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) If it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        # coeff ranges 0..1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # Determine the device type based on the input string.
    def _device_type(self, device):
        lower_string = device.lower()
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


class BlockDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(
            self.data[idx : idx + self.block_size].astype(np.int64)  # noqa
        )  # noqa
        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.block_size].astype(  # noqa E203
                np.int64
            )  # noqa
        )  # noqa
        return x, y


class GPTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, block_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.global_step = 0
        super().__init__()

    def setup(self, stage):
        self.train_data, self.val_data = self._get_dataset(self.data_dir)
        self.train_sampler = ElasticDistributedSampler(self.train_data)

    def train_dataloader(self):
        dataset = BlockDataset(self.train_data, self.block_size)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        dataset = BlockDataset(self.val_data, self.block_size)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=4)

    def state_dict(self):
        steps_per_epoch = len(self.train_dataloader())
        step_in_epoch = self.global_step % steps_per_epoch
        epoch = int(self.global_step / steps_per_epoch)
        self.train_sampler.set_epoch(epoch)
        state = self.train_sampler.state_dict(step_in_epoch, self.batch_size)
        return state

    def load_state_dict(self, state_dict) -> None:
        self.train_sampler.load_state_dict(state_dict)

    def _get_dataset(self, data_dir):
        train_data = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        val_data = np.memmap(
            os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
        )
        return train_data, val_data


def log_rank0(msg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(msg)


def setup(args):
    global local_rank

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if use_cuda:
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"rank {rank} is initialized local_rank = {local_rank}")
    # This process will do logging, checkpointing etc.
    rank == 0
    seed_offset = rank  # Each process gets a different seed
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn


def cleanup():
    dist.destroy_process_group()


def train(args):
    global local_rank
    data_module = GPTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )

    # train
    model = Nanogpt(args, ctx=None)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        every_n_train_steps=1000,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )

    callbacks = [UpdateDataStepCallback(), checkpoint_callback]

    trainer = pl.Trainer(
        max_steps=args.max_iters,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        callbacks=callbacks,
    )

    trainer.fit(model, data_module)


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
    parser.add_argument("--batch_size", type=int, default=16, required=False)
    parser.add_argument("--block_size", type=int, default=128, required=False)

    # Model settings
    parser.add_argument("--checkpoint_dir", type=str, required=False)
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
    args = arg_parser()
    setup(args)
    train(args)
    cleanup()
