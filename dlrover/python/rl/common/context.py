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
import threading
from typing import Dict, Tuple

from omegaconf import DictConfig

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.common.singleton import Singleton
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.enums import (
    RLAlgorithmType,
    RLRoleType,
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
        self._trainer_arc_type: TrainerArcType = TrainerArcType[
            trainer_arc_type
        ]
        self._algorithm_type: RLAlgorithmType = RLAlgorithmType[algorithm_type]
        self._config: DictConfig = config

    def __repr__(self):
        return (
            f"Trainer(class={self._module_class}, "
            f"type={self._trainer_type}, "
            f"arc_type={self._trainer_arc_type}, "
            f"algorithm_type={self._algorithm_type}, "
            f"config={self._config})"
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
        trainer: TrainerDesc,
        workloads: Dict[RLRoleType, WorkloadDesc],
    ):
        """
        Description of reinforcement learning's computing architecture.

        Args:
            trainer: The description for the trainer.
            workloads: A dictionary of workloads, including: actor_workload,
                generator_workload, ref_workload, reward_workload,
                critic_workload.
        """

        self._trainer = trainer
        self._workloads = workloads

    def __repr__(self):
        return (
            f"RLContext({self._trainer}, "
            f"actor:{self.actor_workload}, "
            f"generator:{self.generator_workload}, "
            f"reference:{self.ref_workload}, "
            f"reward:{self.reward_workload}, "
            f"critic:{self.critic_workload})"
        )

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
    def generator_workload(self):
        return self._workloads[RLRoleType.GENERATOR]

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
                {
                    RLRoleType.ACTOR: actor_desc,
                    RLRoleType.GENERATOR: generator_desc,
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
        # trainer
        if not self.trainer:
            logger.error("Trainer not set.")
            return False
        else:
            if (
                not self.trainer.module_name
                or not self.trainer.class_name
                or not self.trainer.config
            ):
                logger.error(
                    "Trainer mandatory arguments: module, class or "
                    "config has empty value."
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

        # actor
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

        return True


class JobContext(Singleton, PickleSerializable):
    """
    JobContext includes all the key runtime information.
    """

    def __init__(self):
        self._job_config = None
        self._rl_context = None

        self._locker = threading.Lock()

    def init(self, job_config: JobConfig, rl_context: RLContext):
        self._job_config = job_config
        self._rl_context = rl_context

    @property
    def job_config(self) -> JobConfig:
        return self._job_config

    @property
    def rl_context(self) -> RLContext:
        return self._rl_context


def get_job_context() -> JobContext:
    job_context = JobContext.singleton_instance()
    return job_context
