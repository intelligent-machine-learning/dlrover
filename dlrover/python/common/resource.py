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
from typing import Dict


class Resource(object):
    def __init__(self, cpu=0, memory=0, disk=0, gpu=0, ud_resource=None):
        self._cpu: float = cpu
        self._memory: int = memory
        self._disk: int = disk
        self._gpu: float = gpu

        self._ud_resource: Dict[str, float] = ud_resource

    def __repr__(self):
        return (
            f"Resource(cpu={self._cpu}, "
            f"mem={self._memory}, "
            f"disk={self._disk}, "
            f"gpu={self._gpu}, "
            f"user_defined={self._ud_resource})"
        )

    @property
    def cpu(self):
        return self._cpu

    @property
    def memory(self):
        return self._memory

    @property
    def disk(self):
        return self._disk

    @property
    def gpu(self):
        return self._gpu

    @property
    def ud_resource(self):
        return self._ud_resource

    @classmethod
    def default_cpu(cls):
        return Resource(cpu=1)

    @classmethod
    def default_gpu(cls):
        return Resource(gpu=1)

    @classmethod
    def from_dict(cls, resource_dict):
        if not resource_dict:
            return Resource()

        cpu_value = 0
        mem_value = 0
        disk_value = 0
        gpu_value = 0
        ud_value = {}

        for key, value in resource_dict.items():
            if key.lower() == "cpu":
                cpu_value = value
            elif key.lower() == "memory" or key.lower() == "mem":
                mem_value = value
            elif key.lower() == "disk":
                disk_value = value
            elif key.lower() == "gpu":
                gpu_value = value
            else:
                ud_value[key] = value

        return cls(
            cpu=cpu_value,
            memory=mem_value,
            disk=disk_value,
            gpu=gpu_value,
            ud_resource=ud_value,
        )

    def validate(self) -> bool:
        if self.cpu < 0 or self.memory < 0 or self.disk < 0 or self.gpu < 0:
            return False
        return True
