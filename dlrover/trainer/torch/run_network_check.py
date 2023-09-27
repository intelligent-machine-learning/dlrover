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
import json
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from dlrover.python.common.constants import ConfigPath
from dlrover.python.common.log import default_logger as logger

FAULT_CHECK_TASK = "fault-check"
STRAGGLER_CHECK_TASK = "straggler-check"


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
    elapsed_time = time.time() - start
    return elapsed_time


def matmul(use_cuda, round=10):
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    tensor1 = torch.randn(10, 2048, 1024).to(device)
    tensor2 = torch.randn(10, 1024, 2048).to(device)

    start = int(time.time())
    for _ in range(round):
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


def main(task):
    use_cuda = torch.cuda.is_available()
    start_init = time.time()
    if use_cuda:
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    init_time = round(time.time() - start_init, 3)
    task_time = 0
    if task == FAULT_CHECK_TASK:
        shape = 1 << 20
        task_time = bm_all_gather(shape, use_cuda)
    elif task == STRAGGLER_CHECK_TASK:
        task_time = matmul(use_cuda)
        shape = 1 << 24
        task_time += bm_all_gather(shape, use_cuda)
    local_rank = int(os.environ["LOCAL_RANK"])
    elapsed_time = round(init_time + task_time, 3)
    write_time_to_file(elapsed_time, local_rank)
    if local_rank == 0:
        logger.info(
            f"Init process group costs {init_time}."
            f"Execution costs {task_time}s"
        )
    return elapsed_time


def arg_parser():
    parser = argparse.ArgumentParser(description="Network checker")
    parser.add_argument(
        "--task",
        type=str,
        default=STRAGGLER_CHECK_TASK,
        choices=[FAULT_CHECK_TASK, STRAGGLER_CHECK_TASK],
        required=False,
    )
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    main(args.task)
    logger.info("Finish testing machine.")
