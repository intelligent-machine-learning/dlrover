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
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union

import ray
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    field_validator,
    model_validator,
)
from ray.actor import ActorClass
from typing_extensions import TypeAlias

from dlrover.python.unified.common.enums import WorkloadEntrypointType
from dlrover.python.unified.util.actor_helper import as_actor_class
from dlrover.python.util.common_util import get_class_by_module_and_class_name


def get_entrypoint_type(entry_point: str) -> Optional[WorkloadEntrypointType]:
    entry_point = entry_point.strip()

    if (
        entry_point.endswith(".py")
        or "/" in entry_point
        or entry_point.startswith("./")
    ):
        return WorkloadEntrypointType.PY_CMD
    parts = entry_point.split()
    if parts and parts[0].endswith(".py"):
        return WorkloadEntrypointType.PY_CMD

    if re.fullmatch(
        r"^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+$", entry_point
    ):
        return WorkloadEntrypointType.MODULE_FUNC

    return None


class ResourceDesc(BaseModel):
    cpu: float = Field(default=0.0, ge=0)
    memory: int = Field(
        default=0, ge=0, validation_alias=AliasChoices("memory", "mem")
    )
    disk: int = Field(default=0, ge=0)
    accelerator: float = Field(
        default=0.0,
        ge=0,
        validation_alias=AliasChoices("accelerator", "acc", "gpu"),
    )
    user_defined: Dict[str, float] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("user_defined", "ud_resource"),
    )

    @classmethod
    def get_or_default(cls, resource: Dict[str, Union[int, float]]):
        desc = cls.model_validate(resource)

        if desc.is_empty():
            return ResourceDesc(accelerator=1)
        return desc

    def is_empty(self) -> bool:
        """
        Check if the resource description is empty.
        """
        return (
            self.cpu == 0.0
            and self.accelerator == 0.0
            and not self.user_defined
        )

    def __add__(self, other: "ResourceDesc") -> "ResourceDesc":
        assert isinstance(other, ResourceDesc), (
            f"Cannot add {type(other)} to ResourceDesc."
        )
        user_defined = self.user_defined.copy()
        for k, v in other.user_defined.items():
            user_defined[k] = user_defined.get(k, 0.0) + v
        return ResourceDesc(
            cpu=self.cpu + other.cpu,
            memory=self.memory + other.memory,
            disk=self.disk + other.disk,
            accelerator=self.accelerator + other.accelerator,
            user_defined=user_defined,
        )

    def __mul__(self, factor: float) -> "ResourceDesc":
        user_defined = {k: v * factor for k, v in self.user_defined.items()}
        return ResourceDesc(
            cpu=self.cpu * factor,
            memory=int(self.memory * factor),
            disk=int(self.disk * factor),
            accelerator=self.accelerator * factor,
            user_defined=user_defined,
        )


class BaseWorkloadDesc(BaseModel, ABC):
    """
    Base description of a workload.
    """

    total: int = Field(
        default=1,
        ge=0,  # allow 0 to indicate no instances,
        validation_alias=AliasChoices("total", "num", "number"),
    )
    resource: ResourceDesc = Field(
        default_factory=ResourceDesc,
        validation_alias=AliasChoices("resource", "res"),
    )
    envs: Dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("env", "environment", "envs"),
    )
    per_node_max_failure: int = Field(
        default=3,
        description="The maximum limit of failures count in a single node. "
        "Will relaunch the corresponding node if this limit exceeded.",
    )
    max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of single workload restarts.",
    )
    group: Optional[str] = Field(
        default=None,
        description="The name of the workload group this workload belongs to."
        "If not specified, each workload will be placed in its own group.",
    )
    per_group: int = Field(
        default=1,
        ge=1,
        description="The number of this workload instances "
        "per workload group.",
    )
    entry_point: str = Field(
        description="The entry point for the workload in `module.func` pattern or `command`(xxx.py arg0 arg1) pattern",
        validation_alias=AliasChoices("entry_point", "entrypoint"),
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="The configuration for the workload. "
        "This is used to pass additional parameters to the workload.",
    )

    rank_based_gpu_selection: bool = Field(
        default=False,
        description=(
            "If True, GPUs are selected according to the local rank and visible devices. "
            "Otherwise, GPU allocation is managed by Ray and only the allocated GPUs are made visible."
        ),
    )
    is_driver: bool = Field(
        default=True,
        description=(
            "If True, job will wait for this workload to complete. "
            "If False, job will not wait, suitable for RPC driven workloads."
        ),
    )

    @model_validator(mode="after")
    def validate(self):
        assert self.total % self.per_group == 0, (
            f"instance_number {self.total} must be divisible by "
            f"per_group {self.per_group}."
        )
        if self.rank_based_gpu_selection:
            assert self.per_group > 1 and self.resource.accelerator <= 1, (
                "rank_based_gpu_selection is only valid when "
                "per_group > 1 and resources(gpu) <= 1."
            )
        return self

    @field_validator("entry_point", mode="before")
    def _normalize_entry_point(cls, entry_point):
        if isinstance(entry_point, str):
            return entry_point.strip().replace("::", ".")
        return entry_point

    @field_validator("entry_point")
    def _validate_entry_point(cls, entry_point):
        if not get_entrypoint_type(entry_point):
            raise ValueError("Invalid entrypoint.")
        return entry_point

    @field_validator("resource", mode="after")
    def _require_resource_not_empty(cls, resource: ResourceDesc):
        if resource.is_empty():
            raise ValueError(
                "Resource must not be empty. "
                "Please specify at least one of cpu, accelerator, or custom."
            )
        return resource

    @abstractmethod
    def get_worker_cls(self) -> ActorClass: ...

    def get_master_cls(self) -> Optional[ActorClass]:
        return None

    @property
    def entry_point_type(self):
        return get_entrypoint_type(self.entry_point)


class ElasticWorkloadDesc(BaseWorkloadDesc):
    """
    Description of an elastic workload.
    """

    backend: Literal["elastic"] = Field(default="elastic")
    comm_pre_check: bool = Field(
        default=True,
        description="Whether to perform communication pre_check before starting the workload.",
    )
    comm_auto_setup_process_group: bool = Field(
        default=True,
        description="Whether to automatically setup process group for communication. "
        "Or just set MASTER_ADDR and MASTER_PORT envs.",
    )
    comm_backend: str = Field(
        default="auto",
        description="Communication backend for the elastic workload. "
        "Supported backends: 'auto', 'gloo', 'nccl', 'mpi'.",
    )
    comm_timeout_s: Optional[int] = Field(
        default=None,
        ge=0,
        description="Timeout for communication operations in seconds.",
    )
    # TODO node_min,max,unit when supporting scaling
    # TODO numa_affinity,exclude_straggler,network_check,comm_perf_test,auto_tunning,save_at_breakpoint,

    def get_worker_cls(self) -> ActorClass:
        from dlrover.python.unified.backend import ElasticWorker

        return as_actor_class(ElasticWorker)

    def get_master_cls(self) -> ActorClass:
        from dlrover.python.unified.backend.elastic.master import ElasticMaster

        return as_actor_class(ElasticMaster)


class SimpleWorkloadDesc(BaseWorkloadDesc):
    """Description of a simple workload.
    This is used for workloads that do not require elastic scaling.
    """

    backend: Optional[Literal["simple"]] = Field(default="simple")

    def get_worker_cls(self) -> ActorClass:
        from dlrover.python.unified.backend.common.base_worker import (
            BaseWorker,
        )

        return as_actor_class(BaseWorker)


class CustomWorkloadDesc(BaseWorkloadDesc):
    """
    Description of a workload with custom backend.

    Not recommended to use this workload unless other options are not suitable.
    """

    backend: Literal["custom"] = Field(default="custom")
    module_name: str = Field(alias="module_name")
    class_name: str = Field(alias="class_name")
    # Not used for custom workload, kept for compatibility
    entry_point: str = ""

    def get_worker_cls(self) -> ActorClass:
        cls = get_class_by_module_and_class_name(
            self.module_name, self.class_name
        )
        from ray.actor import ActorClass

        if not isinstance(cls, ActorClass):
            # raise TypeError(f"Class {self.class_name} is not an ActorClass.")
            return ray.remote(cls)  # type: ignore[return-value]
        return cls


# Union type for workload descriptions, discriminating by `kind`.
WorkloadDesc: TypeAlias = Union[
    ElasticWorkloadDesc,
    SimpleWorkloadDesc,
    CustomWorkloadDesc,
]  # type: ignore[valid-type, assignment]
