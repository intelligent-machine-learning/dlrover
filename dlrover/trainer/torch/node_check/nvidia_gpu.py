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

from .utils import (
    bm_allreduce,
    get_network_check_timeout,
    init_process_group,
    matmul,
    record_execution_time,
)


def set_nccl_env():
    env_conf = os.getenv("NCCL_SETTINGS", "")
    if not env_conf:
        return
    for item in env_conf.split(","):
        k, v = item.split("=")
        os.environ[k] = v


@record_execution_time
def main():
    use_cuda = torch.cuda.is_available()
    protocol = "nccl" if use_cuda else "gloo"

    init_process_group(protocol, timeout=get_network_check_timeout())
    if use_cuda:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    t = matmul(use_cuda)
    shape = 1 << 24
    t += bm_allreduce(shape, use_cuda)
    dist.destroy_process_group()
    return t


if __name__ == "__main__":
    set_nccl_env()
    t = main()
