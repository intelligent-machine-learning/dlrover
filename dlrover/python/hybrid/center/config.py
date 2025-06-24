from typing import Any, Dict, List

from pydantic import AliasChoices, BaseModel, Field
from ray.actor import ActorClass

from dlrover.python.common.enums import ResourceType
from dlrover.python.unified.common.constant import DLTrainerConstant
from dlrover.python.unified.common.enums import (
    MasterStateBackendType,
    SchedulingStrategyType,
)
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


class TrainerDesc(BaseModel):
    """
    Description of a trainer.
    """

    # module_name: str = Field(alias="module")
    # class_name: str = Field(alias="class")
    # trainer_type: TrainerType
    node_number: int = 1
    device_type: ResourceType = ResourceType[
        DLTrainerConstant.DEVICE_TYPE_DEFAULT
    ]
    device_per_node: int = DLTrainerConstant.DEVICE_PER_NODE_DEFAULT
    torch_master_port: List[int] = DLTrainerConstant.TORCH_MASTER_PORT_DEFAULT


class WorkloadDesc(BaseModel):
    """
    Description of a workload.
    """

    # module_name: str = Field(alias="module_name")
    # class_name: str = Field(alias="class_name")
    instance_number: int = Field(alias="num")
    per_node: int = Field(default=1)
    instance_resource: ResourceDesc = Field(
        default_factory=ResourceDesc, alias="resource"
    )
    instance_env: Dict[str, str] = Field(default_factory=dict, alias="env")
    config: Dict[str, Any] = Field(default_factory=dict)

    def get_cls(self):
        cls = get_class_by_module_and_class_name(
            self.module_name, self.class_name
        )
        if not isinstance(cls, ActorClass):
            raise TypeError(f"Class {self.class_name} is not an ActorClass.")
        return cls


class DLConfig(BaseModel):
    # config: DictConfig = Field(
    # default_factory=DictConfig,
    # description="The configuration of the deep learning job.",
    # )
    trainer: TrainerDesc = Field(
        default_factory=TrainerDesc,
        description="Description of the trainer.",
    )
    # The module name and class name of the trainer.
    workloads: Dict[str, WorkloadDesc]
    # workload_group: List[Dict[str, float]]  # TODO what is this for?
    global_envs: Dict[str, str] = Field(default_factory=dict)


class JobConfig(BaseModel):
    job_name: str = Field(description="Name of the job.")
    dl_config: DLConfig = Field(
        description="Description of reinforcement learning's computing architecture."
    )
    master_cpu: int = 1  # in cores
    master_mem: int = 128  # in MiB
    master_state_backend_type: MasterStateBackendType = Field(
        default=MasterStateBackendType.RAY_INTERNAL,
        description="The type of the master state backend.",
    )
    master_state_backend_config: Dict = Field(
        default_factory=dict,
        description="The configuration of the master state backend, like: path and so on.",
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
    trainer_max_restart: int = Field(
        default=10,
        description="The maximum limit on the number of trainer restarts.",
    )
    workload_max_restart: Dict[str, int] = Field(
        default_factory=lambda: {"default": 30},
        description="The maximum limit on the number of workload actor restarts.",
    )
