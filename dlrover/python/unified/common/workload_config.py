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

from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union

import ray
from pydantic import AliasChoices, BaseModel, Field, model_validator
from ray.actor import ActorClass
from typing_extensions import TypeAlias

from dlrover.python.unified.util.actor_helper import as_actor_class
from dlrover.python.util.common_util import get_class_by_module_and_class_name


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

    instance_number: int = Field(
        validation_alias=AliasChoices(
            "total", "num", "number", "instance_number"
        )
    )
    instance_resource: ResourceDesc = Field(
        default_factory=ResourceDesc,
        validation_alias=AliasChoices(
            "instance_resource", "resource", "res", "instance_res"
        ),
    )
    instance_env: Dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices(
            "instance_env", "env", "environment", "envs"
        ),
    )
    max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of restarts.",
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

    @model_validator(mode="after")
    def validate(self):
        assert self.instance_number % self.per_group == 0, (
            f"instance_number {self.instance_number} must be divisible by "
            f"per_group {self.per_group}."
        )
        return self

    @abstractmethod
    def get_worker_cls(self) -> ActorClass: ...

    def get_master_cls(self) -> Optional[ActorClass]:
        return None


class ElasticWorkloadDesc(BaseWorkloadDesc):
    """
    Description of an elastic workload.
    """

    backend: Literal["elastic"] = Field(default="elastic")
    entry_point: str = Field(
        description="The entry point for the elastic workload. cls::func"
    )
    comm_backend: str = Field(
        default="gloo",
        description="Communication backend for the elastic workload. "
        "Supported backends: 'gloo', 'nccl', 'mpi'.",
    )
    comm_timeout_s: int = Field(
        default=30,
        ge=0,
        description="Timeout for communication operations in seconds.",
    )
    # TODO node_min,max,unit when supporting scaling
    # TODO numa_affinity,exclude_straggler,network_check,comm_perf_test,auto_tunning,save_at_breakpoint,

    def get_worker_cls(self) -> ActorClass:
        from dlrover.python.unified.backend import ElasticWorker

        return ElasticWorker  # type: ignore[return-value]

    def get_master_cls(self) -> ActorClass:
        from dlrover.python.unified.backend import ElasticMaster

        return as_actor_class(ElasticMaster)  # type: ignore[return-value]


class CustomWorkloadDesc(BaseWorkloadDesc):
    """
    Description of a custom type workload.
    """

    backend: Literal["custom"] = Field(default="custom")
    module_name: str = Field(alias="module_name")
    class_name: str = Field(alias="class_name")
    config: Dict[str, Any] = Field(default_factory=dict)

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
    CustomWorkloadDesc,
]  # type: ignore[valid-type, assignment]
