# Copyright 2024 The DLRover Authors. All rights reserved.
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

import os
import time
import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision


class DummyDataset(Dataset):
    def __init__(self, vocab_size=1000, max_length=128, data_size=100000):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        text = torch.randint(low=0, high=self.vocab_size, size=(self.max_length,))
        return text, text


def main():
    # Initialize the process group
    dist.init_process_group(backend="nccl")

    # Get local rank and world size
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    config = GPTNeoXConfig(
        vocab_size=50257,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=8192,
        max_position_embeddings=2048,
        hidden_act="gelu_new",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        attn_implementation="sdpa",
    )
    model = GPTNeoXForCausalLM(config)

    # Initialize Dummy DataLoader
    dataset = DummyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, sampler=sampler)

    # Wrap Model with FSDP
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={GPTNeoXLayer,},)
    model = FSDP(model, device_id=local_rank, auto_wrap_policy=wrap_policy,
                 mixed_precision=MixedPrecision(param_dtype=torch.float16, cast_forward_inputs=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Training Loop
    epoch = 0
    iters = 0
    while True:
        model.train()
        for input_ids, labels in dataloader:
            start = time.time()
            input_ids, labels = input_ids.to(local_rank), labels.to(local_rank)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if local_rank == 0 and iters % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()} time {time.time() - start}")
            iters += 1
        epoch += 1

    print("Training Complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
