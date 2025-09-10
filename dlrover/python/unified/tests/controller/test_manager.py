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

from unittest.mock import AsyncMock, Mock, patch

import pytest
from pytest_mock import MockerFixture

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.common.enums import MasterStage
from dlrover.python.unified.common.workload_desc import SimpleWorkloadDesc
from dlrover.python.unified.controller import remote_call
from dlrover.python.unified.controller.manager import PrimeManager, logger
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)

# mypy: disable-error-code=method-assign


def test_manager_save_load():
    config = elastic_training_job()
    manager = PrimeManager(config)
    assert PrimeManager.INSTANCE is manager
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
    new_manager._do_stop = Mock()
    new_manager.handle_self_failover()
    assert new_manager.state.stage == MasterStage.READY
    assert new_manager._do_stop.called

    # Case 2. When failover from RUNNING, recover running
    manager._update_stage(MasterStage.RUNNING)
    assert manager.state.stage == MasterStage.RUNNING

    create_main_task = mocker.patch("asyncio.create_task")
    new_manager = PrimeManager(config)
    new_manager.handle_self_failover()
    assert new_manager.state.stage == MasterStage.RUNNING
    assert create_main_task.called


@pytest.mark.parametrize("case", [1, 2, 3])
async def test_manager_handle_actor_restart(mocker: MockerFixture, case):
    config = elastic_training_job()
    config.dl_config.workloads["simple"] = SimpleWorkloadDesc(
        entry_point="simple.run"
    )
    manager = PrimeManager(config)

    # Case 1. Elastic worker restarted
    if case == 1:
        invoke_actor = mocker.patch(
            "dlrover.python.unified.controller.manager.invoke_actor_t",
            AsyncMock(),
        )
        worker = manager.graph.roles["training"].instances[0]
        worker.node_info = NodeInfo("node_1")
        assert worker.role.sub_master is not None

        await manager.deal_with_actor_restarting(worker)
        assert worker.per_node_failure_count == 1
        assert worker.total_failure_count == 1
        assert invoke_actor.called
        assert invoke_actor.call_args[0][0] is remote_call.restart_workers
        assert invoke_actor.call_args[0][1] == worker.role.sub_master.name
    # Case 2. Simple worker restarted
    elif case == 2:
        manager.restart_job = AsyncMock()
        worker = manager.graph.roles["simple"].instances[0]
        worker.node_info = NodeInfo("node_1")

        await manager.deal_with_actor_restarting(worker)
        assert worker.per_node_failure_count == 1
        assert worker.total_failure_count == 1
        assert manager.restart_job.called
    # Case 3. Worker failed cause node relaunch
    elif case == 3:
        manager.restart_job = AsyncMock()  # noop
        worker = manager.graph.roles["simple"].instances[0]
        worker.node_info = NodeInfo("node_1")
        worker.per_node_failure_count = 100  # Large enough

        # Sub 1. relaunch_nodes not implemented
        await manager.deal_with_actor_restarting(worker)
        assert worker.per_node_failure_count == 101
        assert worker.total_failure_count == 1
        assert len(manager.state.removed_nodes) == 0

        # Sub 2. relaunch_nodes
        manager.ext.relaunch_nodes_impl = AsyncMock(
            return_value=[NodeInfo("node_1")]
        )
        await manager.deal_with_actor_restarting(worker)
        assert worker.per_node_failure_count == 102
        assert worker.total_failure_count == 2
        assert worker.node_info.id in manager.state.removed_nodes
        assert manager.ext.relaunch_nodes_impl.called
        assert manager.ext.relaunch_nodes_impl.call_args[0][0] == [
            worker.node_info
        ]

        # Sub 3. new worker join due to node relaunch
        await manager.deal_with_actor_restarting(worker)
        assert worker.per_node_failure_count == 0  # no failure
        assert worker.total_failure_count == 2  # keep

        # Sub 4. relaunch_nodes raise exception
        manager.ext.relaunch_nodes_impl = AsyncMock(side_effect=Exception())
        manager.state.removed_nodes = set()
        worker.per_node_failure_count = 100  # Large enough
        await manager.deal_with_actor_restarting(worker)
        assert worker.node_info.id not in manager.state.removed_nodes


async def test_some_misc_cases(mocker: MockerFixture):
    config = elastic_training_job()
    manager = PrimeManager(config)

    # request_stop in READY stage
    manager.state.stage = MasterStage.READY
    manager._do_stop = Mock()
    manager.request_stop("test stop in READY")
    assert manager._do_stop.called

    # save exception
    mocker.patch.object(
        manager.state_backend, "set", side_effect=Exception("test")
    )
    with patch.object(
        logger, "exception", side_effect=logger.exception
    ) as mock_log:
        manager.save()
    assert mock_log.called
    assert mock_log.call_args[0][0] == "Failed to save state"

    # _load_state exception
    mocker.patch.object(manager.state_backend, "exists", return_value=True)
    mocker.patch.object(
        manager.state_backend, "get", side_effect=Exception("test")
    )
    with patch.object(
        logger, "exception", side_effect=logger.exception
    ) as mock_log:
        assert manager._load_state() is None
    assert mock_log.called
    assert mock_log.call_args[0][0] == "Failed to load state"


async def test_request_stop_cases():
    config = elastic_training_job()
    manager = PrimeManager(config)

    # Case 1. Actor.restart_count exceeds the limit
    manager.request_stop = Mock()
    worker = manager.graph.roles["training"].instances[0]
    worker.restart_count = worker.spec.max_restart
    await manager.restart_actors([worker])
    assert manager.request_stop.called
    assert worker.name in str(manager.request_stop.call_args[0][0])

    # Case 2. node_restart_count exceeds the limit
    manager.request_stop = Mock()
    manager.state.node_restart_count = config.node_max_restart
    await manager._relaunch_fault_nodes([])
    assert manager.request_stop.called
    assert "node relaunch" in str(manager.request_stop.call_args[0][0])

    # Case 3. job_restart_count exceeds the limit
    manager.request_stop = Mock()
    manager.state.job_restart_count = config.job_max_restart
    manager.state.stage = MasterStage.RUNNING
    manager._task = AsyncMock()

    await manager.restart_job()
    assert manager.request_stop.called
    assert "Job has exceeded" in str(manager.request_stop.call_args[0][0])
