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
from datetime import timedelta

import torch
import torch.distributed as dist

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except Exception:
    torch_npu = None

from .utils import bm_allgather, matmul, record_execution_time


@record_execution_time
def main():
    use_cuda = torch.cuda.is_available()

    protocol = "gloo"
    if use_cuda:  # pragma: no cover
        device = torch.cuda.get_device_name()
        if "Ascend" in device:
            protocol = "hccl"

    dist.init_process_group(protocol, timeout=timedelta(seconds=180))

    if use_cuda:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    t = matmul(use_cuda)
    shape = 1 << 24
    t += bm_allgather(shape, use_cuda)

    if torch_npu:
        torch_npu._npu_shutdown()
    dist.destroy_process_group()
    return t


if __name__ == "__main__":
    t = main()
