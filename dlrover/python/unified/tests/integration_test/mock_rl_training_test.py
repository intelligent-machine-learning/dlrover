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


import pytest

from dlrover.python.unified.api.rl import RLJobBuilder


@pytest.mark.usefixtures("tmp_ray")
def test_mock_rl_training_basic():
    rl_job = (
        RLJobBuilder()
        .node_num(3)
        .device_per_node(2)
        .device_type("CPU")
        .config({"c1": "v1"})
        .global_env({"e0": "v0"})
        .trainer(
            "dlrover.python.unified.tests.test_class",
            "TestInteractiveTrainer",
        )
        .resource(cpu=1)
        .actor("dlrover.python.unified.tests.test_class", "TestActor")
        .total(2)
        .per_group(1)
        .env({"e1": "v1"})
        .rollout("dlrover.python.unified.tests.test_class", "TestRollout")
        .total(2)
        .per_group(1)
        .reference("dlrover.python.unified.tests.test_class", "TestReference")
        .total(2)
        .per_group(1)
        .build()
    )

    rl_job.submit("test", master_cpu=1, master_memory=128)


@pytest.mark.usefixtures("tmp_ray")
def test_mock_rl_training_collocation_all():
    rl_job = (
        RLJobBuilder()
        .node_num(3)
        .device_per_node(1)
        .device_type("CPU")
        .config({"c1": "v1"})
        .global_env({"e0": "v0", "DLROVER_LOG_LEVEL": "DEBUG"})
        .trainer(
            "dlrover.python.unified.tests.test_class",
            "TestInteractiveTrainer",
        )
        .resource(cpu=1)
        .actor("dlrover.python.unified.tests.test_class", "TestActor")
        .total(2)
        .per_group(1)
        .env({"e1": "v1"})
        .rollout("dlrover.python.unified.tests.test_class", "TestRollout")
        .total(2)
        .per_group(1)
        .reference("dlrover.python.unified.tests.test_class", "TestReference")
        .total(2)
        .per_group(1)
        .with_collocation_all()
        .build()
    )

    rl_job.submit("test", master_cpu=1, master_memory=128)


# TODO abnormal test cases
