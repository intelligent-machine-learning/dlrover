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

TORCH_AVAILABLE = True

try:
    import torch
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

addition = """

def _create_dlrover_master_handler(
    params: RendezvousParameters,
) -> RendezvousHandler:
    from dlrover.python.elastic_agent.torch.rdzv_backend import create_backend
    backend, store = create_backend(params)
    return create_handler(store, backend, params)


handler_registry.register(
    "dlrover-master", _create_dlrover_master_handler,
)
"""

if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        exit(0)
    torch_path = os.path.dirname(torch.__file__)

    registry_path = os.path.join(
        torch_path, "distributed/elastic/rendezvous/registry.py"
    )

    with open(registry_path, "a") as f:
        f.write(addition)
