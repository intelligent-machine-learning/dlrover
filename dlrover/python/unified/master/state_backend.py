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
from abc import ABC, abstractmethod

import ray
from ray.experimental.internal_kv import (
    _internal_kv_del,
    _internal_kv_exists,
    _internal_kv_get,
    _internal_kv_list,
    _internal_kv_put,
)

from dlrover.python.unified.common.enums import MasterStateBackendType
from dlrover.python.unified.common.job_context import get_job_context

_job_ctx = get_job_context()


class MasterStateBackendFactory(object):
    @classmethod
    def get_state_backend(cls):
        backend_type = _job_ctx.job_config.master_state_backend_type
        if backend_type == MasterStateBackendType.HDFS:
            # TODO: impl hdfs state backend
            raise NotImplementedError()
        else:
            return RayInternalMasterStateBackend()


class MasterStateBackend(ABC):
    @abstractmethod
    def init(self):
        """Initialize the state-backend."""
        pass

    @abstractmethod
    def get(self, key):
        """Get value by key."""
        pass

    @abstractmethod
    def set(self, key, value):
        """Set value by key."""
        pass

    @abstractmethod
    def delete(self, key):
        """Delete value by key."""
        pass

    @abstractmethod
    def exists(self, key) -> bool:
        """Whether the key exists."""
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Reset all."""
        pass


class RayInternalMasterStateBackend(MasterStateBackend):
    """
    State-backend using ray.experimental.internal_kv.
    """

    def init(self):
        if not ray.is_initialized():
            ray.init()

    def get(self, key):
        return _internal_kv_get(key)

    def set(self, key, value):
        _internal_kv_put(key, value)

    def delete(self, key):
        _internal_kv_del(key)

    def exists(self, key) -> bool:
        return _internal_kv_exists(key)

    def reset(self, prefix):
        keys = _internal_kv_list(prefix)
        for key in keys:
            _internal_kv_del(key)
