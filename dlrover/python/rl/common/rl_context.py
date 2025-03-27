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
from typing import Dict, Tuple

from omegaconf import DictConfig

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.resource import Resource
from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.rl.common.enums import (
    RLAlgorithmType,
    RLRoleType,
    TrainerType,
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


class RLContext(PickleSerializable):
    def __init__(
        self,
        algorithm_type: RLAlgorithmType,
        config: DictConfig,
        trainer: TrainerDesc,
        workloads: Dict[RLRoleType, WorkloadDesc],
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
        """

        self._algorithm_type: RLAlgorithmType = algorithm_type
        self._config: DictConfig = config
        self._trainer = trainer
        self._workloads = workloads

    def __repr__(self):
        return (
            f"RLContext(algorithm_type:{self.algorithm_type}, "
            f"config:{self.config}, "
            f"trainer:{self.trainer}, "
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
    def workloads(self) -> Dict[RLRoleType, WorkloadDesc]:
        return self._workloads

    @property
    def actor_workload(self):
        return self._workloads[RLRoleType.ACTOR]

    @property
    def rollout_workload(self):
        return self._workloads[RLRoleType.ROLLOUT]

    @property
    def ref_workload(self):
        return self._workloads[RLRoleType.REFERENCE]

    @property
    def reward_workload(self):
        return self._workloads[RLRoleType.REWARD]

    @property
    def critic_workload(self):
        return self._workloads[RLRoleType.CRITIC]

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

            actor_desc = None
            rollout_desc = None
            reference_desc = None
            reward_desc = None
            critic_desc = None
            if trainer_desc.trainer_type == TrainerType.OPENRLHF_PPO_DEEPSPEED:
                # TODO
                pass
            else:
                # actor
                wl_conf = conf.get("workload")
                actor_conf = wl_conf.get("actor", None)

                if actor_conf:
                    actor_desc = WorkloadDesc(
                        (actor_conf.get("module"), actor_conf.get("class")),
                        actor_conf.get("num"),
                        actor_conf.get("resource"),
                    )

                # rollout
                rollout_conf = wl_conf.get("rollout", None)
                if rollout_conf:
                    rollout_desc = WorkloadDesc(
                        (
                            rollout_conf.get("module"),
                            rollout_conf.get("class"),
                        ),
                        rollout_conf.get("num"),
                        rollout_conf.get("resource"),
                    )

                # reference
                reference_conf = wl_conf.get("reference", None)
                if reference_conf:
                    reference_desc = WorkloadDesc(
                        (
                            reference_conf.get("module"),
                            reference_conf.get("class"),
                        ),
                        reference_conf.get("num"),
                        reference_conf.get("resource"),
                    )

                # reward
                reward_conf = wl_conf.get("reward", None)
                if reward_conf:
                    reward_desc = WorkloadDesc(
                        (reward_conf.get("module"), reward_conf.get("class")),
                        reward_conf.get("num"),
                        reward_conf.get("resource"),
                    )

                # critic
                critic_conf = wl_conf.get("critic", None)
                if critic_conf:
                    critic_desc = WorkloadDesc(
                        (critic_conf.get("module"), critic_conf.get("class")),
                        critic_conf.get("num"),
                        critic_conf.get("resource"),
                    )
            return RLContext(
                algorithm_type,
                config,
                trainer_desc,
                {
                    RLRoleType.ACTOR: actor_desc,
                    RLRoleType.ROLLOUT: rollout_desc,
                    RLRoleType.REFERENCE: reference_desc,
                    RLRoleType.REWARD: reward_desc,
                    RLRoleType.CRITIC: critic_desc,
                },
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

        return True
