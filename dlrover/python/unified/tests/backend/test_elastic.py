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

import dataclasses
import os
from typing import List
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from dlrover.python.unified.backend.elastic.node_check_manager import (
    NodeCheckManager,
)
from dlrover.python.unified.backend.elastic.worker.worker import (
    ElasticWorker,
    _get_ray_gpu_devices,
)
from dlrover.python.unified.common.actor_base import ActorInfo, JobInfo
from dlrover.python.unified.common.enums import ACCELERATOR_TYPE
from dlrover.python.unified.common.workload_desc import ElasticWorkloadDesc


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


@pytest.mark.parametrize("case", [1, 2, 3])
def test_setup_envs(mocker: MockerFixture, case: int):
    envs = mocker.patch.dict(os.environ, {}, clear=True)
    worker = Mock(ElasticWorker)
    worker.actor_info = ActorInfo(
        "test", "worker", ElasticWorkloadDesc(entry_point="a.b")
    )
    worker.job_info = JobInfo("test", "test", {}, ACCELERATOR_TYPE.GPU)

    # normal GPU
    if case == 1:
        mocker.patch("ray.get_gpu_ids", return_value=[0])
        ElasticWorker._setup_envs(worker)
        assert envs["NAME"] == "test"
        assert envs["LOCAL_RANK"] == "0"
        assert envs["RANK"] == "0"
        assert envs["LOCAL_WORLD_SIZE"] == "1"
        assert envs["WORLD_SIZE"] == "1"
        assert envs["NODE_RANK"] == "0"
        assert envs["ACCELERATE_TORCH_DEVICE"] == "cuda:0"
    # rank_based_gpu_selection
    elif case == 2:
        mocker.patch("torch.cuda.is_available", return_value=True)
        set_device = mocker.patch("torch.cuda.set_device")

        worker.actor_info = dataclasses.replace(
            worker.actor_info,
            local_rank=2,
            rank=2,
            spec=ElasticWorkloadDesc(
                entry_point="a.b",
                total=4,
                per_group=4,
                rank_based_gpu_selection=True,
            ),
        )
        ElasticWorker._setup_envs(worker)
        assert envs["ACCELERATE_TORCH_DEVICE"] == "cuda:2"
        assert set_device.call_count == 1
        assert str(set_device.call_args[0][0]) == "cuda:2"
    # CPU
    elif case == 3:
        worker.job_info = dataclasses.replace(
            worker.job_info, accelerator_type=ACCELERATOR_TYPE.CPU
        )
        ElasticWorker._setup_envs(worker)
        assert envs["ACCELERATE_TORCH_DEVICE"] == "cpu"
