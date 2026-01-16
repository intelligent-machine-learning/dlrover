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

import os
import tempfile
from unittest.mock import MagicMock, patch

from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.common.actor_base import (
    ActorBase,
    ActorInfo,
    NodeInfo,
)
from dlrover.python.unified.common.enums import ExecutionResultType
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)


def test_get_node_info():
    config = elastic_training_job()
    job_info = config.to_job_info()

    actor_info = ActorInfo(
        name="test_actor",
        role="worker",
        spec=config.dl_config.workloads["training"],
    )

    mock_runtime_context = MagicMock()
    mock_runtime_context.node_id.hex.return_value = "test-node-id"
    mock_runtime_context.was_current_actor_reconstructed = False

    with (
        patch("ray.is_initialized", return_value=True),
        patch("ray.get_runtime_context", return_value=mock_runtime_context),
        patch(
            "dlrover.python.common.env_utils.get_hostname_and_ip",
            return_value=("ray-host", "10.0.0.1"),
        ),
        patch(
            "dlrover.python.unified.common.actor_base.init_main_loop",
            return_value=None,
        ),
        patch.dict("os.environ", {"RAY_ENV": "ray_value"}),
    ):
        actor = ActorBase(job_info, actor_info)
        node_info = actor.get_node_info()

        assert isinstance(node_info, NodeInfo)
        assert node_info.id == "test-node-id"
        assert node_info.hostname == "ray-host"
        assert node_info.ip_address == "10.0.0.1"
        assert node_info.envs.get("RAY_ENV") == "ray_value"


def test_execute_user_command_with_system_exit():
    config = elastic_training_job()
    job_info = config.to_job_info()

    actor_info = ActorInfo(
        name="test_worker_exit",
        role="worker",
        spec=config.dl_config.workloads["training"],
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write("""
import sys
print("Exiting with code 0")
sys.exit(0)
""")
        test_script = f.name

    try:
        mock_runtime_context = MagicMock()
        mock_runtime_context.node_id.hex.return_value = "test-node-id"
        mock_runtime_context.was_current_actor_reconstructed = False

        with (
            patch("ray.is_initialized", return_value=True),
            patch(
                "ray.get_runtime_context", return_value=mock_runtime_context
            ),
            patch(
                "dlrover.python.common.env_utils.get_hostname_and_ip",
                return_value=("ray-host", "10.0.0.1"),
            ),
            patch(
                "dlrover.python.unified.common.actor_base.init_main_loop",
                return_value=None,
            ),
            patch.dict("os.environ", {"RAY_ENV": "ray_value"}),
            patch(
                "dlrover.python.unified.controller.api.PrimeMasterApi.report_execution_result"
            ) as mock_report,
        ):
            worker = BaseWorker(job_info, actor_info)
            worker._execute_user_command(test_script)

            # Verify success result was reported for exit code 0
            mock_report.assert_called_once()
            args = mock_report.call_args[0]
            assert args[0] == "test_worker_exit"
            assert args[1].result == ExecutionResultType.SUCCESS
    finally:
        os.unlink(test_script)


def test_execute_user_command_with_exception():
    config = elastic_training_job()
    job_info = config.to_job_info()

    actor_info = ActorInfo(
        name="test_worker_fail",
        role="worker",
        spec=config.dl_config.workloads["training"],
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write("""
raise ValueError("Test exception in user command")
""")
        test_script = f.name

    try:
        mock_runtime_context = MagicMock()
        mock_runtime_context.node_id.hex.return_value = "test-node-id"
        mock_runtime_context.was_current_actor_reconstructed = False

        with (
            patch("ray.is_initialized", return_value=True),
            patch("ray.get_actor", return_value=MagicMock()),
            patch(
                "ray.get_runtime_context", return_value=mock_runtime_context
            ),
            patch(
                "dlrover.python.common.env_utils.get_hostname_and_ip",
                return_value=("ray-host", "10.0.0.1"),
            ),
            patch(
                "dlrover.python.unified.common.actor_base.init_main_loop",
                return_value=None,
            ),
            patch.dict("os.environ", {"RAY_ENV": "ray_value"}),
            patch(
                "dlrover.python.unified.controller.api.PrimeMasterApi.report_execution_result"
            ) as mock_report,
        ):
            worker = BaseWorker(job_info, actor_info)
            worker._execute_user_command(test_script)

            # Verify failure result was reported
            mock_report.assert_called_once()
            args = mock_report.call_args[0]
            assert args[0] == "test_worker_fail"
            assert args[1].result == ExecutionResultType.FAIL
    finally:
        os.unlink(test_script)
