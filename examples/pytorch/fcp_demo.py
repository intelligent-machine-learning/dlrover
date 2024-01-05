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

"""
The demo demonstrates how to use Flash Checkpoint in a DDP job.
We can start a DDP job by

```
pip install dlrover[torch] -U
dlrover-run --max_restarts=2 --nproc_per_node=2 fcp_demo.py
```
"""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from dlrover.trainer.torch.flash_checkpoint.ddp import (
    DdpCheckpointer,
    StorageType,
)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 16)
        self.fc5 = nn.Linear(16, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    input_dim = 1024
    batch_size = 2048

    device = torch.device("cuda" if use_cuda else "cpu")
    x = torch.rand(batch_size, input_dim).to(device)
    y = torch.rand(batch_size, 1).to(device)

    model = Net(input_dim, 1)
    if use_cuda:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Running basic DDP example on local rank {local_rank}.")
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
    criteria = nn.MSELoss()

    checkpointer = DdpCheckpointer("/tmp/fcp_demo_ckpt")

    # Load checkpoint.
    state_dict = checkpointer.load_checkpoint()
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    if "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])

    step = state_dict.get("step", 0)

    for _ in range(1000):
        step += 1
        predic = model(x)
        loss = criteria(predic, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            # Save checkpoint to memory.
            checkpointer.save_checkpoint(
                step, state_dict, storage_type=StorageType.MEMORY
            )
            print("step {} loss:{:.3f}".format(step, loss))
        if step % 200 == 0:
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            # Save checkpoint to storage.
            checkpointer.save_checkpoint(
                step, state_dict, storage_type=StorageType.DISK
            )
