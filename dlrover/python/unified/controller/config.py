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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, model_validator

from dlrover.python.unified.common.constant import DLTrainerConstant
from dlrover.python.unified.common.enums import MasterStateBackendType
from dlrover.python.unified.common.workload_base import JobInfo
from dlrover.python.unified.common.workload_config import (
    ResourceDesc,
    WorkloadDesc,
)


class ACCELERATOR_TYPE(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    TPU = "TPU"


@dataclass
class WorkloadGroup:
    name: str
    num: int
    workloads: List[Tuple[str, int]]  # (workload, num_instances)
    resource: ResourceDesc


class DLConfig(BaseModel):
    """Description of training configuration.
    This class defines the configurations for algorithm users.
    """

    user_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="User-defined configuration for the deep learning job.",
    )
    workloads: Dict[str, WorkloadDesc]
    global_envs: Dict[str, str] = Field(default_factory=dict)

    node_number: int = 1
    accelerator_type: ACCELERATOR_TYPE = ACCELERATOR_TYPE.GPU
    device_per_node: int = DLTrainerConstant.DEVICE_PER_NODE_DEFAULT

    @property
    def workload_group(self) -> List[WorkloadGroup]:
        """Get the workload groups."""
        groups: Dict[str, WorkloadGroup] = {}
        for name, workload in self.workloads.items():
            group_name = workload.group or f"_group_{name}"
            if group_name not in groups:
                groups[group_name] = WorkloadGroup(
                    name=group_name,
                    num=workload.instance_number // workload.per_group,
                    workloads=[],
                    resource=ResourceDesc(),
                )
            groups[group_name].workloads.append((name, workload.per_group))
            groups[group_name].resource += workload.instance_resource
        # Validate number of instances in each group
        for group in groups.values():
            for name, num in group.workloads:
                workload = self.workloads[name]
                if workload.instance_number != num * workload.per_group:
                    raise ValueError(
                        "Instance number for workload"
                        f" '{name}' is inconsistent.\n  {group}"
                    )
        return list(groups.values())

    @model_validator(mode="after")
    def validate(self):
        for group in self.workload_group:
            if group.resource.accelerator > self.device_per_node:
                raise ValueError(
                    f"Accelerator resource {group.resource.accelerator} "
                    f"exceeds device_per_node {self.device_per_node}."
                )
        sum_accelerator = sum(
            workload.instance_resource.accelerator
            for workload in self.workloads.values()
        )
        if sum_accelerator > self.node_number * self.device_per_node:
            raise ValueError(
                f"Total accelerator resource {sum_accelerator} exceeds "
                f"node_number {self.node_number} * device_per_node "
                f"{self.device_per_node}."
            )
        return self


class JobConfig(BaseModel):
    """Description of all job configuration."""

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
    placement_strategy: str = Field(
        default="SPREAD",
        description="The placement strategy for the job."
        "Refer to ray's placement strategies for more details."
        "Valid options are: STRICT_PACK, STRICT_SPREAD, PACK, SPREAD.",
    )
    job_max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of job-level restarts.",
    )
    master_max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of master restarts.",
    )
    torch_master_port: List[int] = DLTrainerConstant.TORCH_MASTER_PORT_DEFAULT

    def to_job_info(self) -> JobInfo:
        """Convert to JobInfo."""
        return JobInfo(
            name=self.job_name,
            job_id=self.job_name,
            user_config=self.dl_config.user_config,
        )
