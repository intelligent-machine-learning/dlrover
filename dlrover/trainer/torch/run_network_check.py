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


def bm_all_gather(shape, use_cuda):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    data = torch.randn(shape, dtype=torch.float32).to(device)
    tensor_list = [
        torch.zeros_like(data).to(device) for _ in range(world_size)
    ]
    start = int(time.time())
    for _ in range(10):
        dist.all_gather(tensor_list, data)
    end = time.time()
    if local_rank == 0:
        logger.info(f"Test allgather costs {end - start}s")


def main(use_cuda):
    shape = 1 << 20
    bm_all_gather(shape, use_cuda)


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    main(use_cuda)
    logger.info("Finish testing allgather.")
