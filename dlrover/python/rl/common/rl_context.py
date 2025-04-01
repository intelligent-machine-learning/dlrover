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

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.resource import Resource
from dlrover.python.common.serialize import PickleSerializable
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
    ):
        """
        Description of a trainer.

        Args:
            module_class: The module and class of the trainer.
            trainer_type: The trainer type.
        """
        self._module_class: Tuple[str, str] = module_class
        self._trainer_type: TrainerType = TrainerType[trainer_type]

    def __repr__(self):
        return (
            f"Trainer(class={self._module_class}, "
            f"type={self._trainer_type})"
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
    def __init__(self, allocation, unit="process"):
        """
        Description of a workload group(for scheduling).

        Args:
            allocation: The number of the workload instance.
            unit: The allocation unit, default is 'process'.
        """
        self._allocation: Dict[RLRoleType, int] = allocation
        self._unit = unit

    def __repr__(self):
        return (
            f"WorkloadGroupDesc(allocation={self._allocation}, "
            f"unit={self._unit})"
        )

    @classmethod
    def from_dict(cls, dict_value: Dict[str, int]):
        allocation = {}
        for key, value in dict_value.items():
            allocation[RLRoleType[key.upper()]] = value
        return WorkloadGroupDesc(allocation)

    @property
    def allocation(self) -> Dict[RLRoleType, int]:
        return self._allocation

    @property
    def unit(self) -> str:
        return self._unit

    def get_all_roles(self) -> List[RLRoleType]:
        return list(self._allocation.keys())


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
                                WorkloadGroupDesc.from_dict(group_dict)
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
        # algorithm_type
        if not self.algorithm_type:
            logger.error("Algorithm type not set.")
            return False

        # config
        if not self.config:
            logger.error("Config not set.")
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
                self.actor_workload.module_name, self.actor_workload.class_name
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

        # workload group validation
        role_group_definition_num: Dict[RLRoleType, int] = {}
        for group_type, groups in self.workload_groups.items():
            for group_desc in groups:
                for role in group_desc.get_all_roles():
                    if role in role_group_definition_num:
                        role_group_definition_num[role] = (
                            role_group_definition_num[role] + 1
                        )
                    else:
                        role_group_definition_num[role] = 1

        # is same role in different group
        if any(value > 1 for value in role_group_definition_num.values()):
            logger.error(
                "Workload group validation failed: exist repeated role "
                "definition in different group."
            )
            return False

        return True
