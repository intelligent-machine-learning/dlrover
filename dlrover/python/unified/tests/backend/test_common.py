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

import asyncio
import threading
from unittest.mock import Mock

from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.common.workload_base import ActorInfo, JobInfo
from dlrover.python.unified.common.workload_desc import SimpleWorkloadDesc


async def test_start_base():
    info = ActorInfo(
        name="worker1",
        role="worker",
        spec=SimpleWorkloadDesc(entry_point=f"{__name__}._entrypoint"),
    )
    worker = BaseWorker(Mock(JobInfo), info)

    global _entrypoint
    _entrypoint = Mock()

    assert worker.stage == "READY"
    worker.start()
    assert worker.stage == "RUNNING" or worker.stage == "FINISHED"
    while worker.stage != "FINISHED":
        await asyncio.sleep(0)
    assert _entrypoint.call_count == 1


async def test_start_class():
    info = ActorInfo(
        name="worker1",
        role="worker",
        spec=SimpleWorkloadDesc(entry_point=f"{__name__}._entrypoint"),
    )
    worker = BaseWorker(Mock(JobInfo), info)

    init_called = False
    run_called = False

    class entrypoint_class:
        def __init__(self) -> None:
            nonlocal init_called
            init_called = True

        def run(self):
            nonlocal run_called
            assert worker.stage == "RUNNING"
            assert threading.current_thread().name == "user_main_thread"
            run_called = True

    global _entrypoint
    _entrypoint = entrypoint_class

    assert worker.stage == "READY"
    worker.start()
    assert worker.stage == "RUNNING" or worker.stage == "FINISHED"
    assert init_called

    while worker.stage != "FINISHED":
        await asyncio.sleep(0)
    assert run_called
