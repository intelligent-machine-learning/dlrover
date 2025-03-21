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
import pickle
from typing import Dict, Tuple

from omegaconf import DictConfig

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.enums import (
    RLAlgorithmType,
    TrainerArcType,
    TrainerType,
)
from dlrover.python.rl.common.exception import InvalidRLConfiguration
from dlrover.python.util.common_util import get_class_by_module_and_class_name


class TrainerDesc(object):
    def __init__(
        self,
        module_class,
        trainer_type,
        trainer_arc_type,
        algorithm_type,
        config,
    ):
        """
        Description of a trainer.

        Args:
            module_class: The module and class of the trainer.
            trainer_type: The trainer type.
            trainer_arc_type: The trainer architecture type.
            algorithm_type: The algorithm type.
            config: The configuration of the trainer.
        """
        self._module_class: Tuple[str, str] = module_class
        self._trainer_type: TrainerType = TrainerType[trainer_type]
        self._trainer_arc_type: TrainerArcType = TrainerArcType[trainer_arc_type]
        self._algorithm_type: RLAlgorithmType = RLAlgorithmType[algorithm_type]
        self._config: DictConfig = config

    def __str__(self):
        return f"Trainer(class={self._module_class}, type={self._trainer_type}, arc_type={self._trainer_arc_type}, algorithm_type={self._algorithm_type}, config={self._config})"

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
    def trainer_arc_type(self) -> TrainerArcType:
        return self._trainer_arc_type

    @property
    def algorithm_type(self) -> RLAlgorithmType:
        return self._algorithm_type

    @property
    def config(self) -> DictConfig:
        return self._config


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
            self._resource: Dict[str, float] = resource
        else:
            self._resource: Dict[str, float] = {}

    def __str__(self):
        return f"Workload(class={self._module_class}, num={self._num}, resource={self._resource})"

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


class RLContext(object):
    def __init__(
        self,
        trainer: TrainerDesc,
        actor_workload: WorkloadDesc,
        generator_workload: WorkloadDesc,
        ref_workload: WorkloadDesc,
        reward_workload: WorkloadDesc,
        critic_workload: WorkloadDesc,
    ):
        """
        Description of a workload.

        Args:
            trainer: The description for the trainer.
            actor_workload: The description for the actor workload.
            generator_workload: The description for the generator workload.
            ref_workload: The description for the reference workload.
            reward_workload: The description for the reward workload.
            critic_workload: The description for the critic workload.
        """

        self._trainer = trainer
        self._actor_workload = actor_workload
        self._generator_workload = generator_workload
        self._ref_workload = ref_workload
        self._reward_workload = reward_workload
        self._critic_workload = critic_workload

    def __str__(self):
        return f"RLContext({self._trainer}, actor:{self._actor_workload}, generator:{self._generator_workload}, reference:{self._ref_workload}, reward:{self._reward_workload}, critic:{self._critic_workload})"

    @property
    def trainer(self):
        return self._trainer

    @property
    def actor_workload(self):
        return self._actor_workload

    @property
    def generator_workload(self):
        return self._generator_workload

    @property
    def ref_workload(self):
        return self._ref_workload

    @property
    def reward_workload(self):
        return self._reward_workload

    @property
    def critic_workload(self):
        return self._critic_workload

    def serialize(self):
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data) -> "RLContext":
        return pickle.loads(data)

    @classmethod
    def build_from_args(cls, args):
        conf: DictConfig = args.rl_config
        if not conf:
            raise InvalidRLConfiguration()

        try:
            # trainer
            trainer_desc = TrainerDesc(
                (conf.get("module"), conf.get("class")),
                conf.get("type"),
                conf.get("arc_type"),
                conf.get("algorithm_type"),
                conf.get("config"),
            )

            actor_desc = None
            generator_desc = None
            reference_desc = None
            reward_desc = None
            critic_desc = None
            if trainer_desc.trainer_type == TrainerType.OPENRLHF_DEEPSPEED:
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

                # generator
                generator_conf = wl_conf.get("generator", None)
                if generator_conf:
                    generator_desc = WorkloadDesc(
                        (
                            generator_conf.get("module"),
                            generator_conf.get("class"),
                        ),
                        generator_conf.get("num"),
                        generator_conf.get("resource"),
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
                trainer_desc,
                actor_desc,
                generator_desc,
                reference_desc,
                reward_desc,
                critic_desc,
            )
        except Exception as e:
            logger.error(
                f"Got invalid arguments while building RLContext: str{e}"
            )
            raise InvalidRLConfiguration()

    def validate(self) -> bool:
        # trainer
        if not self.trainer:
            logger.error("Trainer not set.")
            return False
        else:
            if not self.trainer.module_name or not self.trainer.class_name or not self.trainer.config:
                logger.error("Trainer mandatory arguments: module, class or "
                             "config has empty value.")
                return False
            if not get_class_by_module_and_class_name(self.trainer.module_name, self.trainer.class_name):
                logger.error("Trainer not found "
                             f"by module {self.trainer.module_name} "
                             f"and class {self.trainer.class_name}.")
                return False

        # actor
        if not self.actor_workload:
            logger.error("Actor workload not set.")
            return False
        else:
            if not self.actor_workload.module_name or not self.actor_workload.class_name:
                logger.error("Actor workload mandatory arguments: module or "
                             "class has empty value.")
                return False
            if not get_class_by_module_and_class_name(self.actor_workload.module_name, self.actor_workload.class_name):
                logger.error("Actor workload not found "
                             f"by module {self.actor_workload.module_name} "
                             f"and class {self.actor_workload.class_name}.")
                return False

        return True
