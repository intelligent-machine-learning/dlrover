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

from dlrover.python.unified.common.enums import MasterStage
from dlrover.python.unified.controller.manager import PrimeManager
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)


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
