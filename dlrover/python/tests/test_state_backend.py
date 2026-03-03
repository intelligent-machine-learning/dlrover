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
import unittest

from dlrover.python.util.state.store_mananger import StoreManager


class StoreManagerTest(unittest.TestCase):
    def test_memory_store_mananger(self):
        os.environ["state_backend_type"] = "Memory"
        store_manager = StoreManager("jobname", "state", "config")
        self.assertEqual(store_manager.store_type(), None)
        memory_store_manager = store_manager.build_store_manager()
        self.assertEqual(memory_store_manager.store_type(), "Memory")
        memory_store = memory_store_manager.build_store()
        memory_store.put("key", "value")
        self.assertEqual(memory_store.get("key"), "value")
        memory_store.delete("key")
        self.assertEqual(memory_store.get("key"), None)
