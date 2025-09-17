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

try:
    import torch_musa  # noqa: F401
except ImportError:
    print("torch_musa is not available")
    pass


def patch_after_import_torch():
    if not hasattr(torch, "musa"):
        return
    # 1. Patch for torch.xxx
    torch.cuda.is_available = torch.musa.is_available
    torch.cuda.current_device = lambda: f"musa:{torch.musa.current_device()}"
    torch.cuda.device_count = torch.musa.device_count
    torch.cuda.set_device = torch.musa.set_device
    torch.cuda.DoubleTensor = torch.musa.DoubleTensor
    torch.cuda.FloatTensor = torch.musa.FloatTensor
    torch.cuda.LongTensor = torch.musa.LongTensor
    torch.cuda.HalfTensor = torch.musa.HalfTensor
    torch.cuda.BFloat16Tensor = torch.musa.BFloat16Tensor
    torch.cuda.IntTensor = torch.musa.IntTensor
    torch.cuda.synchronize = torch.musa.synchronize
    torch.cuda.empty_cache = torch.musa.empty_cache
    torch.Tensor.cuda = torch.Tensor.musa
    torch.cuda.Event = torch.musa.Event
    torch.cuda.current_stream = torch.musa.current_stream
    torch.cuda.get_device_properties = torch.musa.get_device_properties
    if hasattr(torch.musa, "is_bf16_supported"):
        torch.cuda.is_bf16_supported = torch.musa.is_bf16_supported
    else:
        # Fallback for older versions of torch_musa
        torch.cuda.is_bf16_supported = lambda: False

    # retain torch.empty reference
    original_empty = torch.empty

    # redifine torch.empty
    def patched_empty(*args, **kwargs):
        if "device" in kwargs and kwargs["device"] == "cuda":
            kwargs["device"] = "musa"
        result = original_empty(*args, **kwargs)
        return result

    torch.empty = patched_empty


patch_after_import_torch()
