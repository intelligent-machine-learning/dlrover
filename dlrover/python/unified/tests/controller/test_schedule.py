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
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlrover.python.unified.common.config import (
    ACCELERATOR_TYPE,
    DLConfig,
    JobConfig,
)
from dlrover.python.unified.common.workload_desc import (
    ElasticWorkloadDesc,
    ResourceDesc,
    SimpleWorkloadDesc,
)
from dlrover.python.unified.controller.schedule import scheduler
from dlrover.python.unified.controller.schedule.graph import DLExecutionGraph
from dlrover.python.unified.controller.schedule.scheduler import Scheduler


@pytest.fixture
def demo_config():
    return DLConfig(
        workloads={
            "actor": SimpleWorkloadDesc(
                total=4,
                entry_point="module.FakeActor",
                resource=ResourceDesc(cpu=0.31),
                group="actor_rollout",
            ),
            "reference": SimpleWorkloadDesc(
                total=4,
                entry_point="module.FakeReference",
                resource=ResourceDesc(cpu=0.32),
            ),
            "rollout": SimpleWorkloadDesc(
                total=4,
                entry_point="module.FakeRollout",
                resource=ResourceDesc(cpu=0.33),
                group="actor_rollout",
            ),
            "elastic": ElasticWorkloadDesc(
                total=4,
                entry_point="fake_train.run",
            ),
        },
        accelerator_type=ACCELERATOR_TYPE.CPU,
    )


@pytest.fixture
def tmp_scheduler(demo_config: DLConfig) -> Scheduler:
    return Scheduler(
        JobConfig(
            job_name="test_scheduler",
            dl_config=demo_config,
        )
    )


def test_graph(demo_config: DLConfig):
    graph = DLExecutionGraph.create(demo_config)
    assert graph is not None
    assert len(graph.roles) == len(demo_config.workloads)
    for name, workload in demo_config.workloads.items():
        role_in_graph = graph.roles[name]
        assert role_in_graph.spec == workload
        assert role_in_graph.name == name
        assert role_in_graph.instance_number == workload.total
        if name == "elastic":
            assert role_in_graph.sub_master is not None
        else:
            assert role_in_graph.sub_master is None

        assert len(role_in_graph.instances) == role_in_graph.instance_number
        for instance in role_in_graph.instances:
            assert instance.role == role_in_graph
            assert instance.spec == workload
            assert instance.world_size == role_in_graph.instance_number
        assert [instance.rank for instance in role_in_graph.instances] == list(
            range(role_in_graph.instance_number)
        )
    assert len(graph.vertices) == 16 + 1  # 14 workers + 1 elastic sub-master
    assert len(graph.by_name) == 16 + 1  # 14 workers + 1 elastic sub-master


def test_allocate_placement_group(tmp_scheduler: Scheduler):
    tmp_scheduler._create_pg = MagicMock()  # type:ignore[method-assign]
    graph = DLExecutionGraph.create(tmp_scheduler._config.dl_config)

    tmp_scheduler.allocate_placement_group(graph)
    assert tmp_scheduler._create_pg.call_count == 1

    bundles: List[ResourceDesc] = tmp_scheduler._create_pg.call_args[0][0]
    assert len(bundles) == 12  # 3 groups: actor_rollout, reference, elastic

    ref_role = graph.roles["reference"]
    for worker in ref_role.instances:
        assert worker.bundle_index >= 0
        assert bundles[worker.bundle_index].cpu == ref_role.spec.resource.cpu
    for i in range(4):
        assert (
            graph.roles["actor"].instances[i].bundle_index
            == graph.roles["rollout"].instances[i].bundle_index
        ), "Group shared the same bundle"


def test_create_actors(tmp_scheduler: Scheduler):
    scheduler.wait_ready = AsyncMock()  # type:ignore[method-assign]
    tmp_scheduler._create_pg = MagicMock()  # type:ignore[method-assign]
    tmp_scheduler.create_actor = MagicMock()  # type:ignore[method-assign]
    graph = DLExecutionGraph.create(tmp_scheduler._config.dl_config)

    tmp_scheduler.allocate_placement_group(graph)
    asyncio.run(tmp_scheduler.create_actors(graph))
    assert tmp_scheduler.create_actor.call_count == len(graph.vertices)
