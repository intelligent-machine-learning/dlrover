from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union

import ray
from pydantic import AliasChoices, BaseModel, Field
from ray.actor import ActorClass
from typing_extensions import TypeAlias

from dlrover.python.hybrid.util.actor_helper import as_actor_class
from dlrover.python.util.common_util import get_class_by_module_and_class_name


class ResourceDesc(BaseModel):
    cpu: float = Field(default=0.0)
    memory: int = Field(
        default=0, validation_alias=AliasChoices("memory", "mem")
    )
    disk: int = Field(default=0)
    gpu: float = Field(default=0.0)
    gpu_type: str = Field(default="")
    user_defined: Dict[str, float] = Field(
        default_factory=dict, alias="ud_resource"
    )

    def is_empty(self) -> bool:
        """
        Check if the resource description is empty.
        """
        return self.cpu == 0.0 and self.gpu == 0.0 and not self.user_defined


class BaseWorkloadDesc(BaseModel, ABC):
    """
    Base description of a workload.
    """

    instance_number: int = Field(alias="num")
    per_node: int = Field(default=1)
    instance_resource: ResourceDesc = Field(
        default_factory=ResourceDesc, alias="resource"
    )
    instance_env: Dict[str, str] = Field(default_factory=dict, alias="env")

    @abstractmethod
    def get_worker_cls(self) -> ActorClass: ...

    def get_master_cls(self) -> Optional[ActorClass]:
        return None


class ElasticWorkloadDesc(BaseWorkloadDesc):
    """
    Description of an elastic workload.
    """

    kind: Literal["elastic"] = Field(default="elastic")
    cmd: str = Field(description="Command to run the elastic workload.")

    def get_worker_cls(self) -> ActorClass:
        from dlrover.python.hybrid.elastic.worker import ElasticWorker

        return ElasticWorker  # type: ignore[return-value]

    def get_master_cls(self) -> ActorClass:
        from dlrover.python.hybrid.elastic.master import ElasticMaster

        return as_actor_class(ElasticMaster)  # type: ignore[return-value]


class OtherWorkloadDesc(BaseWorkloadDesc):
    """
    Description of a non-elastic workload.
    """

    kind: Literal["other"] = Field(default="other")
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
WorkloadDesc: TypeAlias = Union[ElasticWorkloadDesc, OtherWorkloadDesc]
