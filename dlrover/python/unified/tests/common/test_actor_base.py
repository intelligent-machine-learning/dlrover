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

from unittest.mock import MagicMock, patch

from dlrover.python.unified.common.actor_base import (
    ActorBase,
    ActorInfo,
    NodeInfo,
)
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
