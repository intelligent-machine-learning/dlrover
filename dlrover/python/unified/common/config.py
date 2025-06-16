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
from typing import Dict

from pydantic import BaseModel, Field

from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.common.dl_context import DLContext
from dlrover.python.unified.common.enums import (
    MasterStateBackendType,
    SchedulingStrategyType,
)
from dlrover.python.unified.master.state_backend import MasterStateBackend


class JobConfig(BaseModel):
    job_name: str = Field(description="Name of the job.")
    dl_context: DLContext = Field(
        description="Description of reinforcement learning's computing architecture."
    )
    master_state_backend_type: MasterStateBackendType = Field(
        description="The type of the master state backend."
    )
    master_state_backend_config: Dict = Field(
        description="The configuration of the master state backend, like: path and so on."
    )
    scheduling_strategy_type: SchedulingStrategyType = Field(
        description="The type of scheduling strategy to create workloads."
    )
    job_max_restart: int = Field(
        default=10, description="The maximum limit on the number of job-level restarts."
    )
    master_max_restart: int = Field(
        default=10, description="The maximum limit on the number of master restarts."
    )
    trainer_max_restart: int = Field(
        default=10, description="The maximum limit on the number of trainer restarts."
    )
    workload_max_restart: Dict[str, int] = Field(
        default_factory=lambda: {"default": 30},
        description="The maximum limit on the number of workload actor restarts.",
    )

    def get_workload_max_restart(self, role: str):
        role = role.upper()

        if role in self.workload_max_restart:
            return max(self.workload_max_restart[role], self.job_max_restart)
        return DLMasterConstant.WORKLOAD_MAX_RESTART

    @classmethod
    def build_from_args(cls, args):
        return JobConfig.model_validate(args)

    def create_state_backend(self) -> MasterStateBackend:
        backend_type = self.master_state_backend_type
        if backend_type == MasterStateBackendType.HDFS:
            # TODO: impl hdfs state backend
            raise NotImplementedError()
        else:
            from dlrover.python.unified.master.state_backend import (
                RayInternalMasterStateBackend,
            )

            return RayInternalMasterStateBackend()
