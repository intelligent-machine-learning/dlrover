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
from typing import List

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.master.master import BaseMaster
from dlrover.python.unified.master.mpmd.failover import MPMDFailoverCoordinator
from dlrover.python.unified.master.mpmd.job_manager import MPMDJobManager


@ray.remote
class MPMDMaster(BaseMaster):
    """
    MPMD master is a control implementation designed for training scenarios
    involving multiple compute roles.
    """

    def __init__(self, job_config_serialized, dl_context_serialized):
        super().__init__(job_config_serialized, dl_context_serialized)

        self._failover_coordinator = None
        self.init()
        logger.info(f"MPMD master initialized: {self.job_name}.")

    @property
    def failover_coordinator(self):
        return self._failover_coordinator

    def init(self):
        self._job_manager = MPMDJobManager()
        self._failover_coordinator = MPMDFailoverCoordinator(
            self._job_manager, self._save_context_to_checkpoint, self.exit_job
        )

    def _handle_failures(self, failures: List[FailureDesc]):
        self.failover_coordinator.handle_failures(failures)

    """Remote call functions start"""
    # TODO

    """Remote call functions end"""
