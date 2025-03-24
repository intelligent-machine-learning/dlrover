# Copyright 2025 The EasyDL Authors. All rights reserved.
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
from abc import ABC

import ray
from ray.actor import ActorHandle


class BaseWorkload(ABC):
    def __init__(self, master_handle: ActorHandle):
        self._master_handle = master_handle

    @property
    def master_handle(self) -> ActorHandle:
        return self._master_handle

    def report_master(self):
        ray.get(self._master_handle.report.remote())
