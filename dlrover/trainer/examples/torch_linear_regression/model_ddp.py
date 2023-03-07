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

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_DDP(backend="gloo", verbose=False):
    """
    We don't set ADDR and PORT in here, like:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
    Because program's ADDR and PORT can be given automatically at startup.
    E.g. You can set ADDR and PORT by using:
        python -m torch.distributed.launch
          --master_addr="192.168.1.201" --master_port=23456 ...

    You don't set rank and world_size in dist.init_process_group() explicitly.

    :param backend:
    :param verbose:
    :return:
    """
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # If the OS is Windows or macOS, use gloo instead of nccl
    dist.init_process_group(backend=backend)
    # set distributed device
    if verbose:
        print(
            f"local rank: {local_rank}, \
              global rank: {rank}, world size: {world_size}"
        )
    return rank, local_rank, world_size, None


def cleanup():
    dist.destroy_process_group()


rank, local_rank, world_size, device = setup_DDP(verbose=True)


x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def demo_basic(rank=None, world_size=None):

    # create model and move it to GPU with id rank
    model = LinearModel()
    model = DDP(model)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(100):
        optimizer.zero_grad()
        outputs = model(x_data)
        loss_fn(outputs, y_data).backward()
        optimizer.step()
        print(outputs)
    cleanup()


if __name__ == "__main__":
    demo_basic()
