from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union

import ray
from pydantic import BaseModel, Field
from ray.actor import ActorClass
from typing_extensions import TypeAlias

from dlrover.python.hybrid.center.config import ResourceDesc
from dlrover.python.util.common_util import get_class_by_module_and_class_name


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

    kind: Literal["elastic"]
    cmd: str = Field(description="Command to run the elastic workload.")

    def get_worker_cls(self) -> ActorClass:
        from dlrover.python.hybrid.elastic.worker import ElasticWorker

        return ray.remote(ElasticWorker)  # type: ignore[return-value]

    def get_master_cls(self) -> ActorClass:
        from dlrover.python.hybrid.elastic.master import ElasticMaster

        return ray.remote(ElasticMaster)  # type: ignore[return-value]


class OtherWorkloadDesc(BaseWorkloadDesc):
    """
    Description of a non-elastic workload.
    """

    kind: Literal["other"]
    module_name: str = Field(alias="module_name")
    class_name: str = Field(alias="class_name")
    config: Dict[str, Any] = Field(default_factory=dict)

    def get_worker_cls(self) -> ActorClass:
        cls = get_class_by_module_and_class_name(
            self.module_name, self.class_name
        )
        from ray.actor import ActorClass

        if not isinstance(cls, ActorClass):
            raise TypeError(f"Class {self.class_name} is not an ActorClass.")
        return cls


# Union type for workload descriptions, discriminating by `kind`.
WorkloadDesc: TypeAlias = Union[ElasticWorkloadDesc, OtherWorkloadDesc]
