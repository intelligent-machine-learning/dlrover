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

import pytest
from pytest_mock import MockerFixture

from dlrover.python.unified.common.actor_base import NodeInfo
import dlrover.python.unified.util.node_helper as nh


@pytest.mark.timeout(1, func_only=True)
@pytest.mark.asyncio
async def test_wait_ray_node_relaunching(mocker: MockerFixture):
    ray_node = mocker.patch("ray.nodes")
    ray_node.return_value = [
        {"NodeID": "node1", "Alive": True},
        {"NodeID": "node2", "Alive": True},
    ]

    async def remove_node():
        await asyncio.sleep(0.2)
        ray_node.return_value[1]["Alive"] = False

    async def readd_node():
        await asyncio.sleep(1)
        ray_node.return_value[1]["Alive"] = True

    bg_0 = asyncio.create_task(remove_node())
    bg_1 = asyncio.create_task(readd_node())
    await nh.wait_ray_node_relaunching(["node2"], 2, interval=0.1)
    await bg_0
    await bg_1


def test_get_node_group(mocker: MockerFixture):
    mock_ray_nodes = [
        {
            "NodeID": "node-1",
            "Labels": {"group": "worker-group-1", "zone": "zone-a"},
        },
        {
            "NodeID": "node-2",
            "Labels": {"group": "worker-group-1", "zone": "zone-a"},
        },
        {
            "NodeID": "node-3",
            "Labels": {"group": "worker-group-2", "zone": "zone-b"},
        },
        {"NodeID": "node-4", "Labels": {}},
    ]

    mocker.patch("ray.nodes", return_value=mock_ray_nodes)

    # Test case 1: Node with group label exists and has matching nodes
    node = NodeInfo(id="node-1")
    result = nh.get_node_group(node, "group")
    assert len(result) == 2
    assert {n.id for n in result} == {"node-1", "node-2"}

    # Test case 2: Node with group label exists but no matching nodes
    node = NodeInfo(id="node-3")
    result = nh.get_node_group(node, "zone")
    assert len(result) == 1
    assert result[0].id == "node-3"

    # Test case 3: Node without group label returns single node
    node = NodeInfo(id="node-4")
    result = nh.get_node_group(node, "nonexistent")
    assert len(result) == 1
    assert result[0].id == "node-4"

    # Test case 4: Node doesn't exist in ray.nodes
    node = NodeInfo(id="node-999")
    result = nh.get_node_group(node, "group")
    assert len(result) == 1
    assert result[0].id == "node-999"
