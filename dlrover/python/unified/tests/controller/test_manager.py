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

from unittest.mock import MagicMock, patch, AsyncMock

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
from dlrover.python.unified.util.actor_helper import BatchInvokeResult


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

    actor0 = DLExecutionWorkerVertex(
        role=test_role,
        rank=0,
        node_rank=0,
        world_size=2,
        local_rank=0,
        local_world_size=1,
        restart_count=2,
    )
    actor1 = DLExecutionWorkerVertex(
        role=test_role,
        rank=1,
        node_rank=1,
        world_size=2,
        local_rank=0,
        local_world_size=1,
        restart_count=4,
    )
    actors = [actor0, actor1]

    with (
        patch(
            "dlrover.python.unified.controller.manager.invoke_actors_t",
            new_callable=AsyncMock,
        ) as mock_invoke,
        patch("dlrover.python.unified.controller.manager.ray") as mock_ray,
    ):
        mock_invoke.return_value = BatchInvokeResult(
            method_name="get_node_info",
            actors=[actor1.name],
            results=[NodeInfo(id="node1")],
        )

        call_count = 0

        def mock_nodes():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # retrieve current nodes
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
        mock_manager._context.node_total_count = 2
        mock_manager._context.node_restart_count = 0

        await mock_manager.relaunch_nodes_by_actors(actors, wait_interval=1)
        assert actor1.restart_count == 0

        mock_manager._context.node_restart_count = 2
        mock_manager.request_stop = MagicMock()
        await mock_manager.relaunch_nodes_by_actors(actors, wait_interval=1)
        mock_manager.request_stop.assert_called_once()


def test_runtime_context(mock_manager):
    runtime_context = mock_manager._context
    graph = runtime_context.graph
    assert runtime_context is not None

    serialized_context = runtime_context.serialize()
    recover_context = runtime_context.deserialize(serialized_context)

    assert len(recover_context.graph.roles) == len(graph.roles)
    assert len(recover_context.graph.edges) == len(graph.edges)
    assert len(recover_context.graph.vertices) == len(graph.vertices)
    assert recover_context.graph.vertices[1].name == graph.vertices[1].name
