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

import json
import os
import time

import torch
import torch.distributed as dist

from dlrover.python.common.constants import ConfigPath
from dlrover.python.common.log import default_logger as logger


def record_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        t = round(time.time() - start, 2)
        local_rank = int(os.environ["LOCAL_RANK"])
        write_time_to_file(t, local_rank)
        return t

    return wrapper


def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        t = round(time.time() - start, 2)
        local_rank = int(os.environ["LOCAL_RANK"])
        func_name = func.__name__
        logger.info(
            f"Time to execute {func_name} on local rank {local_rank} is {t}s."
        )

    return wrapper


def mock_error():
    if os.environ.get("MOCK_ERR_RANK"):
        err_rank = int(os.environ["MOCK_ERR_RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if err_rank == local_rank:
            raise ValueError("Mock network error!")


@log_execution_time
def bm_all_gather(shape, use_gpu):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    data = torch.randn(shape, dtype=torch.float32).to(device)
    tensor_list = [
        torch.zeros_like(data).to(device) for _ in range(world_size)
    ]

    dist.barrier()
    for _ in range(10):
        dist.all_gather(tensor_list, data)
    dist.barrier()
    mock_error()


@log_execution_time
def matmul(use_cuda, round=10):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        size = 2048
    else:
        size = 128
    tensor1 = torch.randn(10, size, size).to(device)
    tensor2 = torch.randn(10, size, size).to(device)

    for _ in range(round):
        torch.matmul(tensor1, tensor2)


def write_time_to_file(time, local_rank):
    data = {"time": time, "local_rank": local_rank}
    root = ConfigPath.NETWORK_CHECK_DATA_DIR
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{local_rank}.txt")
    with open(path, "w") as f:
        f.write(json.dumps(data))
