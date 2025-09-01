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

import pytest

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.common.workload_desc import SimpleWorkloadDesc
from dlrover.python.unified.controller.manager import PrimeManager
from dlrover.python.unified.controller.schedule.graph import (
    DLExecutionWorkerVertex,
    DLWorkloadRole,
)
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)
from dlrover.python.unified.common.enums import MasterStage


@pytest.fixture
def mock_manager(monkeypatch):
    """Create a PrimeManager with mocked ray dependencies."""
    mock_ray = MagicMock()
    mock_ray.nodes.return_value = []
    mock_state_factory = MagicMock()
    mock_state = MagicMock()
    mock_state_factory.get_state_backend.return_value = mock_state
    mock_state.exists = MagicMock(return_value=False)
    mock_state.set = MagicMock(return_value=None)

    # mock module-level ray usage
    monkeypatch.setattr(
        "dlrover.python.unified.controller.manager.ray", mock_ray
    )
    monkeypatch.setattr(
        "dlrover.python.unified.controller.manager.MasterStateBackendFactory",
        mock_state_factory,
    )

    manager = PrimeManager(elastic_training_job())
    manager.ray = mock_ray  # For backward compatibility
    return manager


@pytest.mark.asyncio
async def test_wait_node_relaunch_partial_progress(mock_manager):
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

    mock_manager.ray.nodes.side_effect = mock_nodes

    await mock_manager._wait_node_relaunch(
        relaunch_nodes, current_nodes, wait_interval=0.01
    )

    assert call_count >= 2
    assert mock_manager.ray.nodes.call_count >= 2


@pytest.mark.asyncio
async def test_wait_node_relaunch_timeout_case(mock_manager):
    import asyncio

    current_nodes = [NodeInfo(id="node0")]
    relaunch_nodes = [NodeInfo(id="node0")]

    # mock ray.nodes() to always return the same nodes (simulating timeout)
    mock_manager.ray.nodes.return_value = [{"NodeID": "node0"}]

    # should timeout since no new nodes are added
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            mock_manager._wait_node_relaunch(
                relaunch_nodes, current_nodes, wait_interval=0.01
            ),
            timeout=0.1,
        )

    assert mock_manager.ray.nodes.call_count > 1


@pytest.mark.asyncio
async def test_relaunch_node_if_needed(mock_manager):
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

    with (
        patch("dlrover.python.unified.controller.manager.ray") as mock_ray,
    ):
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

        mock_ray.nodes.side_effect = mock_nodes
        mock_manager.state.node_total_count = 2
        mock_manager.state.node_restart_count = 0

        await mock_manager.relaunch_nodes_by_actors(actors, wait_interval=1)
        assert target_actor.per_node_failure_count == 0


def test_manager_save_load():
    config = elastic_training_job()
    manager = PrimeManager(config)
    assert manager.state.stage == MasterStage.INIT
    manager._update_stage(MasterStage.RUNNING)
    assert manager.state.stage == MasterStage.RUNNING

    new_manager = PrimeManager(config)
    assert new_manager.state.stage == MasterStage.RUNNING
