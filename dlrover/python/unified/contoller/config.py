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

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from dlrover.python.common.enums import ResourceType
from dlrover.python.unified.common.constant import DLTrainerConstant
from dlrover.python.unified.common.enums import (
    MasterStateBackendType,
    SchedulingStrategyType,
)
from dlrover.python.unified.common.workload_base import JobInfo
from dlrover.python.unified.common.workload_config import WorkloadDesc


class DLConfig(BaseModel):
    """Description of training configuration."""

    user_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined configuration for the deep learning job.",
    )
    workloads: Dict[str, WorkloadDesc]
    # workload_group: List[Dict[str, float]]  # TODO what is this for?
    global_envs: Dict[str, str] = Field(default_factory=dict)

    node_number: int = 1
    device_type: ResourceType = ResourceType[
        DLTrainerConstant.DEVICE_TYPE_DEFAULT
    ]
    device_per_node: int = DLTrainerConstant.DEVICE_PER_NODE_DEFAULT
    torch_master_port: List[int] = DLTrainerConstant.TORCH_MASTER_PORT_DEFAULT


class JobConfig(BaseModel):
    """Description of the job configuration."""

    job_name: str = Field(description="Name of the job.")
    dl_config: DLConfig = Field()
    master_cpu: int = 1  # in cores
    master_mem: int = 128  # in MiB
    master_state_backend_type: MasterStateBackendType = Field(
        default=MasterStateBackendType.RAY_INTERNAL,
        description="The type of the master state backend.",
    )
    master_state_backend_config: Dict = Field(
        default_factory=dict,
        description="The configuration of the master state backend, "
        "like: path and so on.",
    )
    scheduling_strategy_type: SchedulingStrategyType = Field(
        default=SchedulingStrategyType.AUTO,
        description="The type of scheduling strategy to create workloads.",
    )
    job_max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of job-level restarts.",
    )
    master_max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of master restarts.",
    )

    def to_job_info(self) -> JobInfo:
        """Convert to JobInfo."""
        return JobInfo(
            name=self.job_name,
            job_id=self.job_name,
            user_config=self.dl_config.user_config,
        )
