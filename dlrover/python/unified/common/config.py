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
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import core_schema

from dlrover.python.unified.common.actor_base import JobInfo
from dlrover.python.unified.common.enums import (
    ACCELERATOR_TYPE,
    MasterStateBackendType,
)
from dlrover.python.unified.common.workload_desc import (
    ResourceDesc,
    WorkloadDesc,
)


@dataclass
class WorkloadGroup:
    name: str
    num: int
    workloads: List[str]
    resource: ResourceDesc


class DLConfig(BaseModel):
    """Description of training configuration.
    This class defines the configurations for algorithm users.
    """

    user_config: Any = Field(
        default_factory=dict,
        description="User-defined configuration for the deep learning job.",
    )
    workloads: Dict[str, WorkloadDesc]
    global_envs: Dict[str, str] = Field(default_factory=dict)

    node_number: int = Field(
        default=1,
        ge=1,
        description="The total number of nodes.",
    )
    accelerator_type: ACCELERATOR_TYPE = ACCELERATOR_TYPE.GPU
    device_per_node: int = Field(
        default=8,
        ge=1,
        description="The number of accelerators per node.",
    )

    @property
    def workload_group(self) -> List[WorkloadGroup]:
        """Get the workload groups."""
        groups: Dict[str, WorkloadGroup] = {}
        for name, workload in self.workloads.items():
            group_name = workload.group or f"_group_{name}"
            if group_name not in groups:
                groups[group_name] = WorkloadGroup(
                    name=group_name,
                    num=workload.total // workload.per_group,
                    workloads=[],
                    resource=ResourceDesc(),
                )
            groups[group_name].workloads.append(name)
            for _ in range(workload.per_group):
                groups[group_name].resource += workload.resource

        # Validate number of instances in each group
        for group in groups.values():
            for name in group.workloads:
                workload = self.workloads[name]
                if workload.total != group.num * workload.per_group:
                    raise ValueError(
                        "Instance number for workload"
                        f" '{name}' is inconsistent.\n  {group}"
                    )
        return list(groups.values())

    @field_validator(
        "user_config",
        mode="plain",
        json_schema_input_type=core_schema.dict_schema(
            core_schema.str_schema(), core_schema.any_schema()
        ),
    )
    def _normalize_user_config(cls, v):
        """Convert None to empty DictConfig."""
        import argparse

        if v is None:
            v = DictConfig({})
        elif isinstance(v, dict):
            v = DictConfig(v)
        elif isinstance(v, DictConfig):
            v = v
        elif isinstance(v, argparse.Namespace):
            v = DictConfig(vars(v))
        else:
            return v  # keep original type for Any
        OmegaConf.resolve(v)
        return v

    @field_serializer("user_config")
    def _serialize_user_config(self, v):
        """Serialize user_config to dict."""
        if isinstance(v, DictConfig):
            return OmegaConf.to_object(v)
        return v

    @field_validator("workloads", mode="after")
    def _require_workloads(cls, workloads):
        """Ensure workloads are defined."""
        if len(workloads) == 0:
            raise ValueError("At least one workload must be defined.")
        return workloads

    @model_validator(mode="after")
    def validate(self):
        for group in self.workload_group:
            if group.resource.accelerator > self.device_per_node:
                raise ValueError(
                    f"Group {group.name} accelerator resource {group.resource.accelerator} "
                    f"exceeds device_per_node {self.device_per_node}."
                )
        sum_accelerator = sum(
            workload.resource.accelerator
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
    master_cpu: int = 2  # in cores
    master_mem: int = Field(
        default=4096,
        description="Memory (in MB) for the master actor.",
        validation_alias=AliasChoices("master_memory", "master_mem"),
    )
    master_create_timeout: float = Field(
        default=600.0,
        description="Timeout for creating the master actor.",
        validation_alias=AliasChoices("master_create_timeout", "timeout"),
    )
    master_state_backend_type: MasterStateBackendType = Field(
        default=MasterStateBackendType.RAY_INTERNAL,
        description="The type of the master state backend.",
    )
    master_state_backend_config: Dict = Field(
        default_factory=dict,
        description="The configuration of the master state backend, "
        "like: path and so on.",
    )
    node_max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of node restarts.",
    )
    job_max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of job-level restarts.",
    )
    master_max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of master restarts.",
    )
    failover_trigger_strategy: int = Field(
        default=2,
        description="When to trigger failover when there is failure. 0: skip, 1: later(will not trigger asap when worker failed), 2: now(trigger asap).",
    )
    failover_exec_strategy: int = Field(
        default=1,
        description="What kind of failover should do when there is failure. 0: no failover, 1: job level failover, 2: role level failover.",
    )
    master_isolation_schedule_resource: str = Field(
        default="",
        description="The master actor's scheduling will use this resource(key:1) if the resource is configured.",
    )
    worker_isolation_schedule_resource: str = Field(
        default="",
        description="The worker actor's scheduling will use this resource(key:1) if the resource is configured.",
    )

    def to_job_info(self) -> JobInfo:
        """Convert to JobInfo."""
        return JobInfo(
            name=self.job_name,
            job_id=self.job_name,
            user_config=self.dl_config.user_config,
            accelerator_type=self.dl_config.accelerator_type,
        )
