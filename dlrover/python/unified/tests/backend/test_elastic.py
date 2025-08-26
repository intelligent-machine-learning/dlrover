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
from typing import List
from unittest.mock import Mock

import pytest

from dlrover.python.unified.backend.elastic.node_check_manager import (
    NodeCheckManager,
)
from dlrover.python.unified.backend.elastic.worker.worker import (
    _get_ray_gpu_devices,
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


def test_get_ray_gpu_devices(mocker):
    mocker.patch("ray.get_gpu_ids", return_value=[])
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    assert [it.type for it in _get_ray_gpu_devices()] == ["cpu"]

    mocker.patch("ray.get_gpu_ids", return_value=[0, 1])
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    assert [it.index for it in _get_ray_gpu_devices()] == [0, 1]

    mocker.patch("ray.get_gpu_ids", return_value=[7, 8])
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8"
    assert [it.index for it in _get_ray_gpu_devices()] == [2, 3]

    with pytest.raises(RuntimeError):
        mocker.patch("ray.get_gpu_ids", return_value=[7, 8])
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
        _ = _get_ray_gpu_devices()
