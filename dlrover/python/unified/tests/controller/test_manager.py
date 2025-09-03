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

from unittest.mock import Mock

import pytest

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.common.enums import MasterStage
from dlrover.python.unified.common.workload_desc import SimpleWorkloadDesc
from dlrover.python.unified.controller.manager import PrimeManager
from dlrover.python.unified.controller.schedule.graph import (
    DLExecutionWorkerVertex,
    DLWorkloadRole,
)
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)


@pytest.mark.asyncio
async def test_wait_node_relaunch_partial_progress(mocker):
    current_nodes = [
        NodeInfo(id="node0"),
        NodeInfo(id="node1"),
        NodeInfo(id="node2"),
    ]
    relaunch_nodes = [NodeInfo(id="node1"), NodeInfo(id="node2")]

    # mock ray.nodes() to simulate gradual node relaunching
    call_count = 0

    def mock_nodes():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # no new nodes relaunched yet
            return [
                {"NodeID": "node0"},
                {"NodeID": "node1"},
                {"NodeID": "node2"},
                {"NodeID": "node3"},
            ]
        elif call_count == 2:
            # node 1 relaunched
            return [
                {"NodeID": "node0"},
                {"NodeID": "node4"},
                {"NodeID": "node2"},
                {"NodeID": "node3"},
            ]
        else:
            # all nodes relaunched
            return [
                {"NodeID": "node0"},
                {"NodeID": "node4"},
                {"NodeID": "node2"},
                {"NodeID": "node5"},
            ]

    manager = PrimeManager(elastic_training_job())
    ray_nodes = mocker.patch("ray.nodes", side_effect=mock_nodes)

    await manager._wait_node_relaunch(
        relaunch_nodes, current_nodes, wait_interval=0.01
    )

    assert call_count >= 2
    assert ray_nodes.call_count >= 2


@pytest.mark.asyncio
async def test_wait_node_relaunch_timeout_case(mocker):
    import asyncio

    current_nodes = [NodeInfo(id="node0")]
    relaunch_nodes = [NodeInfo(id="node0")]

    # mock ray.nodes() to always return the same nodes (simulating timeout)
    ray_nodes = mocker.patch("ray.nodes", return_value=[{"NodeID": "node0"}])

    manager = PrimeManager(elastic_training_job())

    # should timeout since no new nodes are added
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            manager._wait_node_relaunch(
                relaunch_nodes, current_nodes, wait_interval=0.01
            ),
            timeout=0.1,
        )

    assert ray_nodes.call_count > 1


@pytest.mark.asyncio
async def test_relaunch_node_if_needed(mocker):
    spec = SimpleWorkloadDesc(entry_point="test::main")
    test_role = DLWorkloadRole(name="test", spec=spec, instance_number=2)

    target_actor = DLExecutionWorkerVertex(
        role=test_role,
        rank=1,
        node_rank=1,
        world_size=2,
        local_rank=0,
        local_world_size=1,
        restart_count=4,
    )
    target_actor.set_node(NodeInfo(id="node1"))
    actors = [target_actor]

    call_count = 0

    def mock_nodes():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # init nodes
            return [
                {
                    "NodeID": "node0",
                    "NodeManagerHostname": "host0",
                    "NodeManagerAddress": "192.168.1.0",
                },
                {
                    "NodeID": "node1",
                    "NodeManagerHostname": "host1",
                    "NodeManagerAddress": "192.168.1.1",
                },
            ]
        elif call_count == 2:
            # no new nodes relaunched yet
            return [
                {
                    "NodeID": "node0",
                    "NodeManagerHostname": "host0",
                    "NodeManagerAddress": "192.168.1.0",
                },
                {
                    "NodeID": "node1",
                    "NodeManagerHostname": "host1",
                    "NodeManagerAddress": "192.168.1.1",
                },
            ]
        else:
            # node 1 relaunched
            return [
                {
                    "NodeID": "node0",
                    "NodeManagerHostname": "host0",
                    "NodeManagerAddress": "192.168.1.0",
                },
                {
                    "NodeID": "node2",
                    "NodeManagerHostname": "host2",
                    "NodeManagerAddress": "192.168.1.2",
                },
            ]

    ray_nodes = mocker.patch("ray.nodes", side_effect=mock_nodes)
    manager = PrimeManager(elastic_training_job())
    manager.state.node_restart_count = 0

    await manager.relaunch_nodes_by_actors(actors, wait_interval=1)
    assert target_actor.per_node_failure_count == 0


def test_manager_save_load():
    config = elastic_training_job()
    manager = PrimeManager(config)
    assert manager.state.stage == MasterStage.INIT
    manager._update_stage(MasterStage.RUNNING)
    assert manager.state.stage == MasterStage.RUNNING

    new_manager = PrimeManager(config)
    assert new_manager.state.stage == MasterStage.RUNNING


def test_manager_failover(mocker):
    config = elastic_training_job()
    manager = PrimeManager(config)

    # Case 1. When failover from READY, the state should transition to STOPPED
    manager._update_stage(MasterStage.READY)
    assert manager.state.stage == MasterStage.READY

    new_manager = PrimeManager(config)
    new_manager.terminate = Mock()
    new_manager.handle_self_failover()
    assert new_manager.state.stage == MasterStage.READY
    assert new_manager.terminate.called

    # Case 2. When failover from RUNNING, recover running
    manager._update_stage(MasterStage.RUNNING)
    assert manager.state.stage == MasterStage.RUNNING

    create_main_task = mocker.patch("asyncio.create_task")
    new_manager = PrimeManager(config)
    new_manager.handle_self_failover()
    assert new_manager.state.stage == MasterStage.RUNNING
    assert create_main_task.called
