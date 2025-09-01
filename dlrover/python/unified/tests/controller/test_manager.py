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

from unittest.mock import MagicMock

import pytest

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.controller.manager import PrimeManager
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)


@pytest.fixture
def mock_manager(monkeypatch):
    """Create a PrimeManager with mocked ray dependencies."""
    mock_ray = MagicMock()
    mock_ray.nodes.return_value = []

    # mock module-level ray usage
    monkeypatch.setattr(
        "dlrover.python.unified.controller.manager.ray", mock_ray
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
