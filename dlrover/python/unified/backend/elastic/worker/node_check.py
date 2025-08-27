# Copyright 2025 The DLRover Authors. All rights reserved.
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

import torch

from dlrover.trainer.torch.node_check.utils import bm_allreduce, matmul


def run_comm_check() -> float:
    """Run matmul and allreduce to check communication performance."""

    use_cuda = torch.cuda.is_available()

    # warmup
    _ = matmul(use_cuda, round_num=3, verbose=False)

    t = matmul(use_cuda, round_num=500, verbose=True)
    shape = 1 << 24
    t += bm_allreduce(shape, use_cuda)
    return t
