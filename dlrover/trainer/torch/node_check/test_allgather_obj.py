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
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from dlrover.python.common.log import default_logger as logger


def bm_allgather(shape, use_gpu):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    w0 = torch.randn(shape, dtype=torch.float32).to(device)
    w1 = torch.randn(shape, dtype=torch.float32).to(device)
    tensor_list = [None for _ in range(world_size)]

    sd = {"weight_0": w0, "weight_1": w1}

    for i in range(10):
        dist.all_gather_object(tensor_list, sd)

    start = time.time()
    for i in range(10):
        dist.all_gather_object(tensor_list, sd)

    elapsed_time = time.time() - start

    gb_unit = 1024 * 1024 * 1024
    algobw = shape * 4 / gb_unit / (elapsed_time / 1000)
    busbw = algobw * (world_size - 1) / world_size
    algobw = round(algobw, 2)
    busbw = round(busbw, 2)
    elapsed_time = round(elapsed_time, 3)
    if local_rank == 0:
        logger.info(
            f"AllGather Perf: world size = {world_size}, "
            f"algobw={algobw} GB/s, busbw={busbw} GB/s."
        )
    return elapsed_time


def main():
    use_cuda = torch.cuda.is_available()
    protocol = "nccl" if use_cuda else "gloo"

    dist.init_process_group(protocol, timeout=timedelta(seconds=180))
    if use_cuda:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    shape = 1 << 24
    t = bm_allgather(shape, use_cuda)
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.destroy_process_group()
    return t


if __name__ == "__main__":
    t = main()
