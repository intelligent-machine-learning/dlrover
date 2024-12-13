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

# flake8: noqa: E402
import os
import time

import torch

local_rank = int(os.environ.get("LOCAL_RANK", "0"))

print(local_rank)

time.sleep(5)
tensor = torch.ones((10, 10), device=local_rank).bfloat16()

while True:
    time.sleep(0.2)
    torch.cuda.empty_cache()
    res = torch.matmul(tensor, tensor)
    tensor = tensor.cpu()
    tensor = tensor.cuda()
