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
import threading

from dlrover.python.util.state.memory_store import MemoryStore


def state_backend_type():
    backend_type = os.getenv("state_backend_type", "Memory")
    return backend_type


class StoreManager:
    def __init__(self, jobname="", namespace="", config=None):
        self.jobname = jobname
        self.namespace = namespace
        self.config = config

    def build_store_manager(self):
        # state_config = StateConfig(self.config)
        # logger.info(f"Build store manager config : {self.config}")
        # todo :: 这里设计成constant
        if state_backend_type() == "Memory":
            # logger.info("create memory manager")
            return MemoryStoreManager.singleton_instance(
                self.jobname,
                self.namespace,
                self.config,
            )
        else:
            raise RuntimeError(f"No such {state_backend_type()} state backend")

    def store_type(self):
        return None


class MemoryStoreManager(StoreManager):
    _instance_lock = threading.Lock()

    def __init__(self, jobname: str = "", namespace: str = "", config=None):
        super().__init__(jobname, namespace, config)
        self.memory_kv_store = None
        self.jobname = jobname
        self.namespace = namespace
        self.memory_store = None

    def store_type(self):
        return "Memory"

    def build_store(self):
        if self.memory_store is None:
            self.memory_store = MemoryStore(self, self.jobname, "test")
        print(self.memory_store)
        return self.memory_store

    @classmethod
    def singleton_instance(cls, *args, **kwargs):

        if not hasattr(MemoryStoreManager, "_instance"):
            with MemoryStoreManager._instance_lock:
                if not hasattr(MemoryStoreManager, "_instance"):
                    MemoryStoreManager._instance = MemoryStoreManager(
                        *args, **kwargs
                    )
        return MemoryStoreManager._instance


# 增加测试用例
# 测试store
# 测试存储
