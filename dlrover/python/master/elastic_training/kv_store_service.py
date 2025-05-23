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

from threading import Lock
from typing import Dict


class KVStoreService(object):
    def __init__(self):
        self._lock = Lock()
        self._store: Dict[str, bytes] = {}

    def set(self, key, value):
        with self._lock:
            self._store[key] = value

    def get(self, key):
        with self._lock:
            return self._store.get(key, b"")

    def add(self, key, value):
        with self._lock:
            try:
                if key not in self._store:
                    self._store[key] = value
                    return value
                else:
                    v0 = self._store.get(key)
                    self._store[key] = v0 + value
                    return self._store.get(key)
            except Exception:
                return value

    def clear(self):
        self._store.clear()
