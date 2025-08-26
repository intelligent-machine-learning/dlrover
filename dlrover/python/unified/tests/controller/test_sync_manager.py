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

from dlrover.python.unified.controller.sync_manager import SyncManager


def test_register_data_queue():
    manager = SyncManager()
    manager.register_data_queue("test", "test_owner", 2)
    assert len(manager._queues) == 1
    manager.register_data_queue("test", "test_owner", 3)
    assert len(manager._queues) == 1


async def test_get_data_queue_owner():
    manager = SyncManager()
    manager.get_data_queue_owner("test")
    manager.register_data_queue("test", "test_owner", 2)
    owner = await manager.get_data_queue_owner("test")
    assert owner == "test_owner"
