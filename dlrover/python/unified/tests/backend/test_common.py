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
import os
import tempfile
import threading
from unittest.mock import Mock, patch

import pytest

from dlrover.python.unified.api.runtime.rpc_helper import RPC_REGISTRY, rpc
from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.common.actor_base import ActorInfo, JobInfo
from dlrover.python.unified.common.enums import ExecutionResult
from dlrover.python.unified.common.workload_desc import SimpleWorkloadDesc


@rpc()
def module_level_rpc():
    pass


async def test_start_base():
    RPC_REGISTRY.clear()
    info = ActorInfo(
        name="worker1",
        role="worker",
        spec=SimpleWorkloadDesc(entry_point=f"{__name__}._entrypoint"),
    )
    worker = BaseWorker(Mock(JobInfo), info)
    worker._on_execution_end = Mock()

    global _entrypoint
    _entrypoint = Mock()
    _entrypoint.__module__ = __name__

    worker.setup()
    worker.start()
    assert module_level_rpc.__name__ in RPC_REGISTRY

    while not worker._on_execution_end.called:
        await asyncio.sleep(0)
    assert _entrypoint.call_count == 1


async def test_start_class():
    RPC_REGISTRY.clear()
    info = ActorInfo(
        name="worker1",
        role="worker",
        spec=SimpleWorkloadDesc(entry_point=f"{__name__}._entrypoint"),
    )
    worker = BaseWorker(Mock(JobInfo), info)
    worker._on_execution_end = Mock()

    init_called = False
    run_called = False

    class entrypoint_class:
        def __init__(self) -> None:
            nonlocal init_called
            init_called = True

        @rpc()
        def class_level_rpc(self):
            pass

        def run(self):
            nonlocal run_called
            assert run_called is False
            assert threading.current_thread().name == "user_main_thread"
            run_called = True

    global _entrypoint
    _entrypoint = entrypoint_class
    _entrypoint.__module__ = __name__

    worker.setup()
    worker.start()
    assert init_called
    assert entrypoint_class.class_level_rpc.__name__ in RPC_REGISTRY

    while not worker._on_execution_end.called:
        await asyncio.sleep(0)
    assert run_called


@pytest.mark.asyncio
async def test_start_with_py_cmd():
    root_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
        )
    )
    script_path = f"{root_dir}/dlrover/python/unified/tests/integration_test/dummy_run.py"

    RPC_REGISTRY.clear()
    info = ActorInfo(
        name="worker_cmd",
        role="worker",
        spec=SimpleWorkloadDesc(entry_point=f"{script_path} --test 0"),
    )
    worker = BaseWorker(Mock(JobInfo), info)
    worker._on_execution_end = Mock()
    worker.setup()

    with patch("runpy.run_path") as mock_run:
        mock_run.side_effect = lambda path, run_name: None
        worker._start_with_py_cmd(script_path + " --test 0")

        while not worker._on_execution_end.called:
            await asyncio.sleep(0.1)

        mock_run.assert_called_once_with(script_path, run_name="__main__")

    while not worker._on_execution_end.called:
        await asyncio.sleep(0)
    worker._on_execution_end.assert_called_once()


@pytest.mark.asyncio
async def test_start_with_py_cmd_error_handling():
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, "error_script.py")
        with open(script_path, "w") as f:
            f.write("raise ValueError('Test error')")

        info = ActorInfo(
            name="worker_error",
            role="worker",
            spec=SimpleWorkloadDesc(entry_point=script_path),
        )
        worker = BaseWorker(Mock(JobInfo), info)
        worker._on_execution_end = Mock()
        worker.setup()

        # inject error
        with patch("runpy.run_path") as mock_run:
            mock_run.side_effect = ValueError("Test error")
            worker._start_with_py_cmd(script_path)

        while not worker._on_execution_end.called:
            await asyncio.sleep(0.1)
        worker._on_execution_end.assert_called_once_with(ExecutionResult.FAIL)
