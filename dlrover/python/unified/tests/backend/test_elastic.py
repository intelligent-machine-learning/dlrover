from typing import List
from unittest.mock import Mock

from dlrover.python.unified.backend.elastic.node_check_manager import (
    NodeCheckManager,
)
from dlrover.python.unified.common.workload_base import ActorInfo


def test_find_straggling_nodes():
    nodes: List[ActorInfo] = [Mock(ActorInfo) for _ in range(10)]
    node_check = NodeCheckManager()

    res = node_check.find_straggling_nodes(nodes, [0.1, 0.2, 0.3])
    assert len(res) == 0, "No straggling nodes expected"

    res = node_check.find_straggling_nodes(nodes, [0.1, 3, 0.3])
    assert len(res) == 1, "One straggling node expected"
    assert res[0] == nodes[1], "Expected node at index 1 to be straggling"

    res = node_check.find_straggling_nodes(nodes, [0.1, 3, 0.3, 4])
    assert len(res) == 2, "Two straggling nodes expected"
    assert res[0] == nodes[1], "Expected node at index 1 to be straggling"
    assert res[1] == nodes[3], "Expected node at index 3 to be straggling"
