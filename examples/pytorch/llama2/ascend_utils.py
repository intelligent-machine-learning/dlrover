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

try:
    from transformers.utils import is_torch_npu_available
except (ImportError, ModuleNotFoundError):

    def is_torch_npu_available():
        "Checks if `torch_npu` is installed and potentially"
        " if a NPU is in the environment"
        import importlib

        if importlib.util.find_spec("torch_npu") is None:
            return False

        import torch
        import torch_npu  # noqa: F401,F811

        return hasattr(torch, "npu") and torch.npu.is_available()


if is_torch_npu_available():
    import torch_npu  # noqa: F401,F811
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
