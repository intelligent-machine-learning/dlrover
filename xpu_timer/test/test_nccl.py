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
import sys

import numpy as np
import torch
import torch.distributed as dist


def run(rank, size):
    """Distributed function to be implemented later."""
    # group = dist.new_group([0, 1])
    tensors = []
    bufs = []
    for i in range(10):
        if i > 5:
            size = 1024 + i
        else:
            size = 102 + i
        tensor = torch.ones(size, size).cuda().half()
        buf = [torch.zeros(size, size).cuda().half() for _ in range(2)]
        tensors.append(tensor)
        bufs.append(buf)
    a = torch.randn(1024, 1024).cuda().bfloat16()
    b = torch.randn(1024, 1024).cuda().bfloat16()
    c = torch.randn(3, 1024, 1024).cuda().bfloat16()
    d = torch.randn(3, 1024, 1024).cuda().bfloat16()
    count = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    # Waits for everything to finish running

    while True:
        if count % 100 == 0:
            print(f"{count}, {np.array(times).mean()}")
            times = []

        tensor = tensors[count % 10]
        buf = bufs[count % 10]
        dist.all_gather(buf, tensor, async_op=False)
        dist.all_reduce(tensor, async_op=False)
        torch.matmul(a, b)
        start.record()
        torch.matmul(c, d)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        count += 1
        # if count > 1000:
        #   break


def init_process(rank, size, fn, backend="nccl"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29509"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    rank = int(sys.argv[1])
    os.environ["LOCAL_RANK"] = sys.argv[1]
    os.environ["RANK"] = sys.argv[1]
    os.environ["WORLD_SIZE"] = str(size)
    os.environ["LOCAL_WORLD_SIZE"] = str(size)
    init_process(rank, size, run)
