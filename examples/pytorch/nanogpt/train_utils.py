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
import math
import os
import pickle
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from model import GPT, GPTConfig
from torch.utils.data import Dataset

from dlrover.trainer.torch.elastic.dataloader import ElasticDataLoader
from dlrover.trainer.torch.elastic.sampler import ElasticDistributedSampler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class GPTDataset(Dataset):
    def __init__(self, data_path, block_size=128):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(
            self.data[idx : idx + self.block_size].astype(  # noqa E203
                np.int64
            )  # noqa E203
        )  # noqa
        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.block_size].astype(  # noqa E203
                np.int64
            )  # noqa
        )  # noqa
        return x, y


def get_data_loaders(
    data_dir,
    batch_size=32,
    block_size=128,
):
    train_dataset = GPTDataset(os.path.join(data_dir, "train.bin"), block_size)
    val_dataset = GPTDataset(os.path.join(data_dir, "val.bin"), block_size)
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    sampler = ElasticDistributedSampler(dataset=train_dataset)
    train_loader = ElasticDataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=True
    )
    val_loader = ElasticDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    meta_vocab_size = meta["vocab_size"]
    return train_loader, val_loader, meta_vocab_size


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
    log_rank0("Initializing a new model from scratch")
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


def create_lora_config(args):
    if (
        args.lora_rank is None
        and args.lora_dropout is None
        and args.lora_alpha is None
        and args.lora_targets is None
    ):
        return None
    lora_config = {
        "rank": args.lora_rank,
        "dropout": args.lora_dropout,
        "alpha": args.lora_alpha,
        "targets": args.lora_targets.split(",") if args.lora_targets else [],
    }
    return lora_config


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
    rank = int(os.getenv("RANK", 0))
    if rank == 0:
        print(msg)


def setup():
    if torch.cuda.is_available():
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    print(f"rank {rank} is initialized local_rank = {local_rank}")
    # This process will do logging, checkpointing etc.
    seed_offset = rank  # Each process gets a different seed
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # Allow tf32 on cudnn


def cleanup():
    dist.destroy_process_group()


def add_train_args(parser: argparse.ArgumentParser):
    # Data settings
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="out", required=False)
    parser.add_argument(
        "--eval_interval", type=int, default=2000, required=False
    )
    parser.add_argument("--log_interval", type=int, default=1, required=False)
    parser.add_argument("--eval_iters", type=int, default=200, required=False)
    parser.add_argument("--eval_only", action="store_true", required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--block_size", type=int, default=128, required=False)
    parser.add_argument("--epochs", type=int, default=1, required=False)

    # Model settings
    parser.add_argument("--n_layer", type=int, default=6, required=False)
    parser.add_argument("--n_head", type=int, default=6, required=False)
    parser.add_argument("--n_embd", type=int, default=384, required=False)
    parser.add_argument("--dropout", type=float, default=0.0, required=False)
    parser.add_argument("--bias", action="store_true", required=False)

    # LoRA settings
    parser.add_argument("--lora_rank", type=int, default=4, required=False)
    parser.add_argument(
        "--lora_dropout", type=float, default=0.0, required=False
    )
    parser.add_argument(
        "--lora_alpha", type=float, default=1.0, required=False
    )
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="wq,wk,wo,wv",
        required=False,
        help="comma separated list of targets to apply lora to",
    )
    # Optimizer settings
    parser.add_argument(
        "--learning_rate", type=float, default=6e-4, required=False
    )
    parser.add_argument("--max_iters", type=int, default=2000, required=False)
    parser.add_argument(
        "--weight_decay", type=float, default=1e-1, required=False
    )
    parser.add_argument("--beta1", type=float, default=0.9, required=False)
    parser.add_argument("--beta2", type=float, default=0.95, required=False)
    parser.add_argument("--grad_clip", type=float, default=1.0, required=False)
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=0, required=False
    )

    # Learning rate decay settings
    parser.add_argument("--decay_lr", action="store_true", required=False)
    parser.add_argument("--warmup_iters", type=int, default=0, required=False)
    parser.add_argument(
        "--lr_decay_iters", type=int, default=10, required=False
    )
    parser.add_argument("--min_lr", type=float, default=6e-5, required=False)

    parser.add_argument("--compile", type=str, default="False", required=False)
    parser.add_argument(
        "--save_memory_interval", type=int, default=20, required=False
    )
    parser.add_argument(
        "--save_storage_interval", type=int, default=200, required=False
    )
    parser.add_argument(
        "--use_native_ckpt", action="store_true", required=False
    )
    parser.add_argument(
        "--save_dir", type=str, default="/tmp/checkpoint/", required=False
    )
