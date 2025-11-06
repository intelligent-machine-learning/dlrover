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

from dlrover.python.common.log import default_logger as logger

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except Exception:
    torch_npu = None

from .utils import (
    DeviceBenchEnv,
    bm_allgather,
    get_network_check_timeout,
    init_process_group,
    matmul,
    record_execution_time,
)


@record_execution_time
def main():
    use_cuda = torch.cuda.is_available()

    protocol = "gloo"
    if use_cuda:  # pragma: no cover
        device = torch.cuda.get_device_name()
        if "Ascend" in device:
            protocol = "hccl"
    else:
        logger.warning(
            f"Use GLOO as comm protocol for use_cuda: {use_cuda} "
            "or 'Ascend' not in `torch.cuda.get_device_name()`."
        )

    init_process_group(protocol, timeout=get_network_check_timeout())

    if use_cuda:
        os.environ["HCCL_CONNECT_TIMEOUT"] = "120"
        os.environ["HCCL_EXEC_TIMEOUT"] = "180"
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Given that the GPU models on each node are the same, the benchmark
        # environment only needs to be printed once.
        if local_rank == 0:
            device_name = torch.cuda.get_device_name()
            bench_env = DeviceBenchEnv(
                device_name=device_name,
                torch_version=torch.__version__,
                cann_version=torch.version.cann,
            )
            hccl_env = {
                k: v for k, v in os.environ.items() if k.startswith("HCCL")
            }
            logger.info(f"benchmark env: {bench_env}, hccl env: {hccl_env}")

    try:
        # warmup
        _ = matmul(use_cuda, round_num=3, verbose=False)
        result = matmul(use_cuda, round_num=500, verbose=True)
        shape = 1 << 24
        result += bm_allgather(shape, use_cuda)
        return result
    finally:
        dist.destroy_process_group()
        if torch_npu:
            try:
                torch_npu._npu_shutdown()
            except Exception as e:
                logger.warning(f"Got error when cleanup npu: {str(e)}.")


if __name__ == "__main__":
    t = main()
