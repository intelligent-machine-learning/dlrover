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
from typing import Any, ClassVar

from ray.experimental.internal_kv import (
    _internal_kv_del,
    _internal_kv_exists,
    _internal_kv_get,
    _internal_kv_list,
    _internal_kv_put,
)

from dlrover.python.unified.common.enums import MasterStateBackendType
from dlrover.python.unified.util.test_hooks import after_test_cleanup


class MasterStateBackend(ABC):
    @staticmethod
    def create(
        backend_type: MasterStateBackendType, config: Any
    ) -> "MasterStateBackend":
        if backend_type == MasterStateBackendType.RAY_INTERNAL:
            return RayInternalMasterStateBackend()
        elif backend_type == MasterStateBackendType.IN_MEMORY:
            return InMemoryStateBackend()
        elif backend_type == MasterStateBackendType.HDFS:
            # TODO: impl hdfs state backend
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get value by key."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any):
        """Set value by key."""
        pass

    @abstractmethod
    def delete(self, key: str):
        """Delete value by key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether the key exists."""
        pass

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Reset all."""
        pass


class InMemoryStateBackend(MasterStateBackend):
    """State-backend always store nothing"""

    _store: ClassVar[dict] = {}
    after_test_cleanup(_store.clear)

    def get(self, key: str) -> Any:
        return self._store.get(key)

    def set(self, key: str, value: Any):
        self._store[key] = value

    def delete(self, key: str):
        self._store.pop(key, None)

    def exists(self, key: str) -> bool:
        return key in self._store

    def reset(self, *args, **kwargs):
        self._store.clear()


class RayInternalMasterStateBackend(MasterStateBackend):
    """
    State-backend using ray.experimental.internal_kv.
    """

    def get(self, key: str) -> bytes:
        return _internal_kv_get(key)

    def set(self, key: str, value: bytes):
        _internal_kv_put(key, value)

    def delete(self, key: str):
        _internal_kv_del(key)

    def exists(self, key: str) -> bool:
        return _internal_kv_exists(key)

    def reset(self, prefix: str):
        keys = _internal_kv_list(prefix)
        for key in keys:
            _internal_kv_del(key)
