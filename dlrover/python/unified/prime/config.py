from typing import Dict, List

from pydantic import BaseModel, Field

from dlrover.python.common.enums import ResourceType
from dlrover.python.hybrid.common.workload_config import WorkloadDesc
from dlrover.python.unified.common.constant import DLTrainerConstant
from dlrover.python.unified.common.enums import (
    MasterStateBackendType,
    SchedulingStrategyType,
)


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
