# Copyright 2025 The EasyDL Authors. All rights reserved.
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
from typing import Dict, List, Tuple

from omegaconf import DictConfig

from dlrover.python.common.enums import ResourceType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.resource import Resource
from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.rl.common.constant import RLTrainerConstant
from dlrover.python.rl.common.enums import (
    RLAlgorithmType,
    RLRoleType,
    TrainerType,
    WorkloadGroupType,
)
from dlrover.python.rl.common.exception import InvalidRLConfiguration
from dlrover.python.util.common_util import get_class_by_module_and_class_name


class TrainerDesc(object):
    def __init__(
        self,
        module_class,
        trainer_type,
        device_per_node=RLTrainerConstant.DEVICE_PER_NODE_DEFAULT,
        torch_master_port=RLTrainerConstant.TORCH_MASTER_PORT_DEFAULT,
    ):
        """
        Description of a trainer.

        Args:
            module_class (str,str): The module and class of the trainer.
            trainer_type (str): The trainer type. Must be the type of
                'TrainerType'. Default is 'USER_DEFINED'.
            device_per_node (int, optional): How many gpu(cpu) per node.
                Default is 8.
            torch_master_port (int, optional): The port used for torch
                rendzvous(for env 'MASTER_PORT'). Default is 23333.
        """
        self._module_class: Tuple[str, str] = module_class
        self._trainer_type: TrainerType = TrainerType[trainer_type]
        self._device_per_node: int = device_per_node
        self._torch_master_port: int = torch_master_port

    def __repr__(self):
        return (
            f"Trainer(class={self._module_class}, "
            f"type={self._trainer_type}, "
            f"device_per_node={self._device_per_node}, "
            f"torch_master_port={self._torch_master_port})"
        )

    @property
    def module_name(self) -> str:
        return self._module_class[0]

    @property
    def class_name(self) -> str:
        return self._module_class[1]

    @property
    def trainer_type(self) -> TrainerType:
        return self._trainer_type

    @property
    def device_per_node(self):
        return self._device_per_node

    @property
    def torch_master_port(self):
        return self._torch_master_port


class WorkloadDesc(object):
    def __init__(self, module_class, num, resource):
        """
        Description of a workload.

        Args:
            module_class: The module and class of the workload.
            num: The number of the workload instance.
            resource: The resource of the workload instance.
        """
        self._module_class: Tuple[str, str] = module_class
        self._num: int = num
        if resource:
            self._resource: Resource = Resource.from_dict(resource)
        else:
            self._resource: Resource = Resource.default_gpu()

    def __repr__(self):
        return (
            f"Workload(class={self._module_class}, "
            f"num={self._num}, "
            f"resource={self._resource})"
        )

    @property
    def module_name(self) -> str:
        return self._module_class[0]

    @property
    def class_name(self) -> str:
        return self._module_class[1]

    @property
    def instance_number(self):
        return self._num

    @property
    def instance_resource(self):
        return self._resource


class WorkloadGroupDesc(object):
    def __init__(self, group_type, allocation, capacity=0, unit="GPU"):
        """
        Description of a workload group(for scheduling).

        Args:
            group_type: The type of group.
            allocation: The number of the workload instance in group.
            capacity (optional): The resource capacity for each group.
                Default is 0(no limit).
            unit (optional): The resource unit if capacity is specified.
                Default is 'GPU'.
        """
        self._group_type: WorkloadGroupType = group_type
        self._allocation: Dict[RLRoleType, int] = allocation
        self._capacity = capacity
        self._unit: ResourceType = ResourceType[unit.upper()]

    def __repr__(self):
        return (
            f"WorkloadGroupDesc(type={self._group_type}, "
            f"allocation={self._allocation}, "
            f"capacity={self._capacity}, "
            f"unit={self._unit})"
        )

    @classmethod
    def from_dict(cls, group_type, dict_value):
        allocation = {}
        groups = dict_value.get("groups", None)
        if groups:
            for key, value in groups.items():
                allocation[RLRoleType[key.upper()]] = value

        # capacity
        capacity = dict_value.get("capacity", 0)

        # unit
        unit = dict_value.get("unit", "GPU")

        return WorkloadGroupDesc(group_type, allocation, capacity, unit)

    @property
    def group_type(self) -> WorkloadGroupType:
        return self._group_type

    @property
    def allocation(self) -> Dict[RLRoleType, int]:
        return self._allocation

    @property
    def capacity(self):
        if self._capacity <= 0:
            return sum(self._allocation.values())
        return self._capacity

    @property
    def unit(self) -> ResourceType:
        return self._unit

    def is_capacity_limit(self):
        return self._capacity > 0

    def get_all_roles(self) -> List[RLRoleType]:
        return list(self._allocation.keys())

    def has_role(self, role: RLRoleType):
        return role in self.get_all_roles()

    def get_group_name(self) -> str:
        suffix = "_"
        for index, role in enumerate(self.get_all_roles()):
            suffix += role.name
            if index < len(self.get_all_roles()) - 1:
                suffix += "_"
        return self.group_type.name + suffix


class RLContext(PickleSerializable):
    def __init__(
        self,
        algorithm_type: RLAlgorithmType,
        config: DictConfig,
        trainer: TrainerDesc,
        workloads: Dict[RLRoleType, WorkloadDesc],
        workload_groups: Dict[WorkloadGroupType, List[WorkloadGroupDesc]],
    ):
        """
        Description of reinforcement learning's computing architecture.

        Args:
            algorithm_type: The algorithm type.
            config: The full configuration of rl training.
            trainer: The description for the trainer.
            workloads: A dictionary of workloads, including: actor_workload,
                rollout_workload, ref_workload, reward_workload,
                critic_workload.
            workload_groups: The group definition when scheduling the
                different role workloads.
        """

        self._algorithm_type: RLAlgorithmType = algorithm_type
        self._config: DictConfig = config
        self._trainer = trainer
        self._workloads = workloads
        self._workload_groups = workload_groups

    def __repr__(self):
        return (
            f"RLContext(algorithm_type:{self.algorithm_type}, "
            f"config:{self.config}, "
            f"trainer:{self.trainer}, "
            f"group:{self.workload_groups}, "
            f"actor:{self.actor_workload}, "
            f"rollout:{self.rollout_workload}, "
            f"reference:{self.ref_workload}, "
            f"reward:{self.reward_workload}, "
            f"critic:{self.critic_workload})"
        )

    @property
    def algorithm_type(self):
        return self._algorithm_type

    @property
    def config(self):
        return self._config

    @property
    def trainer(self):
        return self._trainer

    @property
    def workload_groups(
        self,
    ) -> Dict[WorkloadGroupType, List[WorkloadGroupDesc]]:
        return self._workload_groups

    @property
    def workloads(self) -> Dict[RLRoleType, WorkloadDesc]:
        return self._workloads

    @property
    def actor_workload(self):
        if RLRoleType.ACTOR in self._workloads:
            return self._workloads[RLRoleType.ACTOR]
        return None

    @property
    def rollout_workload(self):
        if RLRoleType.ROLLOUT in self._workloads:
            return self._workloads[RLRoleType.ROLLOUT]
        return None

    @property
    def ref_workload(self):
        if RLRoleType.REFERENCE in self._workloads:
            return self._workloads[RLRoleType.REFERENCE]
        return None

    @property
    def reward_workload(self):
        if RLRoleType.REWARD in self._workloads:
            return self._workloads[RLRoleType.REWARD]
        return None

    @property
    def critic_workload(self):
        if RLRoleType.CRITIC in self._workloads:
            return self._workloads[RLRoleType.CRITIC]
        return None

    @classmethod
    def build_from_args(cls, args):
        conf: DictConfig = args.rl_config
        if not conf:
            raise InvalidRLConfiguration()

        try:
            algorithm_type = RLAlgorithmType[conf.get("algorithm_type")]
            config = conf.get("config")

            # trainer
            trainer_conf = conf.get("trainer")
            trainer_desc = TrainerDesc(
                (trainer_conf.get("module"), trainer_conf.get("class")),
                trainer_conf.get("type"),
                trainer_conf.get(
                    "device_per_node",
                    RLTrainerConstant.DEVICE_PER_NODE_DEFAULT,
                ),
                trainer_conf.get(
                    "torch_master_port",
                    RLTrainerConstant.TORCH_MASTER_PORT_DEFAULT,
                ),
            )

            # workload
            workload_dict = {}
            wl_conf: DictConfig = conf.get("workload")
            if trainer_desc.trainer_type == TrainerType.OPENRLHF_PPO_DEEPSPEED:
                # TODO: default workload specify
                pass
            else:
                for role_str, workload_desc_dict in wl_conf.items():
                    role_type = RLRoleType[role_str.upper()]
                    workload_desc = WorkloadDesc(
                        (
                            workload_desc_dict.get("module"),
                            workload_desc_dict.get("class"),
                        ),
                        workload_desc_dict.get("num"),
                        workload_desc_dict.get("resource"),
                    )
                    workload_dict[role_type] = workload_desc

            # workload group
            workload_group_dict = {}
            workload_group_conf: DictConfig = conf.get("workload_group", None)
            if workload_group_conf:
                for group_type_str, groups in workload_group_conf.items():
                    try:
                        group_type = WorkloadGroupType[group_type_str.upper()]
                        workload_group_dict[group_type] = []
                        for group_dict in groups:
                            workload_group_dict[group_type].append(
                                WorkloadGroupDesc.from_dict(
                                    group_type, group_dict
                                )
                            )
                    except KeyError:
                        logger.warning(
                            f"Invalid workload group type: {group_type_str}."
                        )
                        continue

            return RLContext(
                algorithm_type,
                config,
                trainer_desc,
                workload_dict,
                workload_group_dict,
            )
        except Exception as e:
            logger.error(
                f"Got invalid arguments while building RLContext: str{e}"
            )
            raise InvalidRLConfiguration()

    def validate(self) -> bool:
        try:
            # config
            if not self.config:
                logger.error("Training config not set.")
                return False

            # trainer
            if not self.trainer:
                logger.error("Trainer not set.")
                return False
            else:
                if not self.trainer.module_name or not self.trainer.class_name:
                    logger.error(
                        "Trainer mandatory arguments: module or class "
                        "has empty value."
                    )
                    return False
                if not get_class_by_module_and_class_name(
                    self.trainer.module_name, self.trainer.class_name
                ):
                    logger.error(
                        "Trainer not found "
                        f"by module {self.trainer.module_name} "
                        f"and class {self.trainer.class_name}."
                    )
                    return False

                # device per node
                if self.trainer.device_per_node <= 0:
                    logger.error("Device per node is invalid.")
                    return False

                # torch master prot
                if self.trainer.torch_master_port <= 0:
                    logger.error("Torch master port is invalid.")
                    return False

            # ====== workload validation ======
            # actor validation
            if not self.actor_workload:
                logger.error("Actor workload not set.")
                return False
            else:
                if (
                    not self.actor_workload.module_name
                    or not self.actor_workload.class_name
                ):
                    logger.error(
                        "Actor workload mandatory arguments: module or "
                        "class has empty value."
                    )
                    return False
                if not get_class_by_module_and_class_name(
                    self.actor_workload.module_name,
                    self.actor_workload.class_name,
                ):
                    logger.error(
                        "Actor workload not found "
                        f"by module {self.actor_workload.module_name} "
                        f"and class {self.actor_workload.class_name}."
                    )
                    return False

            # resource validation
            for role, workload in self.workloads.items():
                if workload and not workload.instance_resource.validate():
                    logger.error(
                        f"Workload {role} resource validation "
                        f"failed: {workload.instance_resource}."
                    )
                    return False

            # ====== workload group validation ======
            role_group_definition_num: Dict[RLRoleType, int] = {}
            for group_type, groups in self.workload_groups.items():
                for group_desc in groups:
                    # collect role definition
                    for role in group_desc.get_all_roles():
                        if role in role_group_definition_num:
                            role_group_definition_num[role] = (
                                role_group_definition_num[role] + 1
                            )
                        else:
                            role_group_definition_num[role] = 1

                    # verify whether the number of instances matches the
                    # number of groups
                    factor = 0
                    for role, num in group_desc.allocation.items():
                        if factor == 0:
                            factor = self.workloads[role].instance_number / num
                        else:
                            if (
                                factor
                                != self.workloads[role].instance_number / num
                            ):
                                logger.error(
                                    "Workload group validation failed: the "
                                    "number of instances does not match the "
                                    "number definition in group."
                                )
                                return False

            # is same role in different group
            if any(value > 1 for value in role_group_definition_num.values()):
                logger.error(
                    "Workload group validation failed: exist repeated role "
                    "definition in different group."
                )
                return False
        except Exception as e:
            logger.error(f"Unexpected error when validate rl context: {e}")
            return False

        return True

    def has_workload_group(self) -> bool:
        return bool(self.workload_groups)
