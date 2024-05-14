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
        t = func(*args, **kwargs)
        local_rank = int(os.environ["LOCAL_RANK"])
        write_time_to_file(t, local_rank)
        return t

    return wrapper


def log_execution_time(func):
    def wrapper(*args, **kwargs):
        t = func(*args, **kwargs)
        t = round(t, 3)
        local_rank = int(os.environ["LOCAL_RANK"])
        func_name = func.__name__
        logger.info(
            f"Time to execute {func_name} on local rank {local_rank} is {t}s."
        )
        return t

    return wrapper


def mock_error():
    if os.environ.get("MOCK_ERR_RANK"):
        err_rank = int(os.environ["MOCK_ERR_RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if err_rank == local_rank:
            raise ValueError("Mock network error!")


@log_execution_time
def bm_allgather(shape, use_gpu):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    data = torch.randn(shape, dtype=torch.float32).to(device)
    tensor_list = [
        torch.zeros_like(data).to(device) for _ in range(world_size)
    ]

    if use_gpu:
        elapsed_time = _execute_nccl_comm(dist.all_gather, tensor_list, data)
    else:
        elapsed_time = _execute_cpu_comm(dist.all_gather, tensor_list, data)

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
    mock_error()
    return elapsed_time


@log_execution_time
def bm_allreduce(shape, use_gpu):
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    data = torch.randn(shape, dtype=torch.float32).to(device)

    if use_gpu:
        elapsed_time = _execute_nccl_comm(dist.all_reduce, data)
    else:
        elapsed_time = _execute_cpu_comm(dist.all_reduce, data)

    gb_unit = 1024 * 1024 * 1024
    algobw = shape * 4 / gb_unit / (elapsed_time / 1000)
    busbw = algobw * 2 * (world_size - 1) / world_size
    algobw = round(algobw, 2)
    busbw = round(busbw, 2)
    elapsed_time = round(elapsed_time, 3)
    if local_rank == 0:
        logger.info(
            f"AllReduce Perf: world size = {world_size}, "
            f"algobw={algobw} GB/s, busbw={busbw} GB/s."
        )
    mock_error()
    return elapsed_time


def _execute_nccl_comm(comm_op, *args):
    local_rank = int(os.environ["LOCAL_RANK"])
    # warm up
    for _ in range(20):
        comm_op(*args)
    torch.cuda.synchronize(device=local_rank)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream=torch.cuda.current_stream())

    round_num = 40
    for _ in range(round_num):
        comm_op(*args)

    end.record(stream=torch.cuda.current_stream())
    end.synchronize()
    elapsed_time = start.elapsed_time(end) / round_num
    return elapsed_time


def _execute_cpu_comm(comm_op, *args):
    # warm up
    for _ in range(10):
        comm_op(*args)

    round_num = 20
    start = time.time()
    for _ in range(round_num):
        comm_op(*args)
    elapsed_time = time.time() - start
    return elapsed_time


@log_execution_time
def matmul(use_cuda, round_num=10):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        size = 2048
    else:
        size = 128
    tensor1 = torch.randn(10, size, size).to(device)
    tensor2 = torch.randn(10, size, size).to(device)

    if use_cuda:
        elapsed_time = _execute_gpu_matmul(tensor1, tensor2, round_num)
    else:
        elapsed_time = _execute_cpu_matmul(tensor1, tensor2, round_num)
    elapsed_time = round(elapsed_time, 3)
    return elapsed_time


def _execute_gpu_matmul(tensor1, tensor2, round_num):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream=torch.cuda.current_stream())

    for _ in range(round_num):
        torch.matmul(tensor1, tensor2)

    end.record(stream=torch.cuda.current_stream())
    end.synchronize()
    elapsed_time = start.elapsed_time(end) / round_num
    return elapsed_time


def _execute_cpu_matmul(tensor1, tensor2, round_num):
    start = time.time()
    for _ in range(round_num):
        torch.matmul(tensor1, tensor2)

    elapsed_time = time.time() - start
    return elapsed_time


def write_time_to_file(time, local_rank):
    data = {"time": time, "local_rank": local_rank}
    root = ConfigPath.NETWORK_CHECK_DATA_DIR
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{local_rank}.txt")
    with open(path, "w") as f:
        f.write(json.dumps(data))
