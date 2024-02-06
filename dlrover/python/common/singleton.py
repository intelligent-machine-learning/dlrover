# Copyright 2022 The DLRover Authors. All rights reserved.
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

import threading
from typing import Optional


def singleton(cls):
    _instance = {}
    _instance_lock = threading.Lock()

    def _singleton(*args, **kwargs):
        with _instance_lock:
            if cls not in _instance:
                _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton


class Singleton(object):
    _instance_lock: Optional[threading.Lock] = None
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def singleton_instance(cls, *args, **kwargs):

        if not cls._instance_lock:
            with cls._lock:
                if not cls._instance_lock:
                    cls._instance_lock = threading.Lock()

        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = cls(*args, **kwargs)
        return cls._instance
