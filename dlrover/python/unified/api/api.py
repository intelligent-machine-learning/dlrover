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
from typing import Dict, List, Set, Tuple, Union

from omegaconf import DictConfig

from dlrover.python.common.constants import CommunicationType, NodeEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import (
    DLWorkloadEnv,
    InternalDLConfig,
    InternalDLWorkloadRole,
)
from dlrover.python.unified.common.enums import (
    DLStreamType,
    DLType,
    TrainerType,
)
from dlrover.python.unified.common.exception import InvalidDLConfiguration
from dlrover.python.unified.driver.main import main


class DLRoleConfig(object):
    def __init__(self, role_name, module_name, class_name, **kwargs):
        self._role_name = role_name
        self._module_name = module_name
        self._class_name = class_name
        self._kwargs = kwargs

    @property
    def role_name(self):
        return self._role_name

    @property
    def module_class(self) -> Tuple[str, str]:
        return self._module_name, self._class_name

    @property
    def others(self):
        return self._kwargs


class DLTrainerConfig(DLRoleConfig):
    """
    Configuration for trainer in task stream.
    """

    def __init__(self, trainer_type, module_name, class_name, **kwargs):
        super().__init__(
            InternalDLWorkloadRole.TRAINER_ROLE,
            module_name,
            class_name,
            **kwargs,
        )
        self._trainer_type: TrainerType = trainer_type

    @property
    def trainer_type(self):
        return self._trainer_type


class DLWorkloadConfig(DLRoleConfig):
    """
    Configuration for all different types' workload.
    """

    def __init__(self, role_name, module_name, class_name, **kwargs):
        super().__init__(role_name, module_name, class_name, **kwargs)

        self._num = 1
        self._per_node = 1
        self._env = {}
        self._sub_stage = []

    @property
    def total(self) -> int:
        return self._num

    @property
    def per_node(self) -> int:
        return self._per_node

    @property
    def env(self) -> dict:
        return self._env

    @property
    def sub_stage(self) -> list:
        return self._sub_stage


class DLJob(object):
    def __init__(
        self,
        dl_type,
        stream_type,
        node_num,
        device_per_node,
        device_type,
        config,
        env,
        components,
        collocations,
    ):
        self._dl_type = dl_type
        self._stream_type = stream_type
        self._node_num = node_num
        self._device_per_node = device_per_node
        self._device_type = device_type
        self._config = config
        self._env = env
        self._components: Dict[str, Union[None, DLWorkloadConfig]] = components
        self._collocations: List[Set[str]] = collocations

    @property
    def dl_type(self):
        return self._dl_type

    @property
    def stream_type(self):
        return self._stream_type

    @property
    def node_num(self):
        return self._node_num

    @property
    def device_per_node(self):
        return self._device_per_node

    @property
    def device_type(self):
        return self._device_type

    @property
    def config(self):
        return self._config

    @property
    def env(self):
        return self._env

    @property
    def collocations(self):
        return self._collocations

    @property
    def trainer(self):
        return self._components[InternalDLWorkloadRole.TRAINER_ROLE]

    def get_workload(self, role):
        return self._components[role]

    def __cal_role_resource(self, role):
        # if role in collocations
        for collocation in self.collocations:
            if role in collocation:
                collocation_process_num = 0
                for role in collocation:
                    collocation_process_num += self._components[role].per_node
                return {
                    self.device_type: round(
                        self.device_per_node / collocation_process_num, 2
                    )
                }

        if role == InternalDLWorkloadRole.ELASTIC_ROLE:
            # elastic agent takes all resource of a node
            return {self.device_type: self.device_per_node}
        else:
            # default
            return {self.device_type: 1}

    def _to_dl_config(self):
        config_result = {}

        # general
        config_result["config"] = self.config
        config_result["env"] = self.env

        # trainer
        trainer_dict = {
            "type": self.trainer.trainer_type.name,
            "module": self.trainer.module_class[0],
            "class": self.trainer.module_class[1],
            "node_number": self.node_num,
            "device_per_node": self.device_per_node,
            "device_type": self.device_type,
        }
        config_result["trainer"] = trainer_dict

        # workload
        workload_dict = {}

        for role, role_config in self._components.items():
            if role == InternalDLWorkloadRole.TRAINER_ROLE:
                continue
            if role_config:
                role_dict = {
                    "module": role_config.module_class[0],
                    "class": role_config.module_class[1],
                    "num": role_config.total,
                    "resource": self.__cal_role_resource(role),
                    "env": role_config.env,
                }
                workload_dict[role] = role_dict

        config_result["workload"] = workload_dict

        # workload group
        workload_group_list = []
        for collocation in self.collocations:
            workload_group = {}
            for role in collocation:
                workload_group[role] = self._components[role].per_node

            workload_group_list.append(workload_group)

        config_result["workload_group"] = workload_group_list

        return config_result

    def submit(
        self,
        job_name,
        blocking=True,
        master_cpu=4,
        master_memory=8192,
        job_max_restart=10,
        **kwargs,
    ):
        """
        Submit the current dl job.

        Args:
            job_name (str): The name of the job.
            blocking (bool, optional): Whether to block until the job is
                complete. Defaults is True.
            master_cpu (int, optional): The number of CPU cores to use.
                Defaults to 4.
            master_memory (int, optional): The number of memory cores to use.
                Unit: mb. Defaults to 8192.
            job_max_restart (int, optional): The maximum number of restarts.
                Defaults to 10.

            Other arguments please refer to:
                'dlrover.python.unified.common.args'
        """

        args = [
            "--job_name",
            job_name,
            "--master_cpu",
            f"{master_cpu}",
            "--master_mem",
            f"{master_memory}",
            "--job_max_restart",
            f"{job_max_restart}",
            "--dl_type",
            f"{self.dl_type}",
            "--dl_config",
            f"{self._to_dl_config()}",
        ]

        for key, value in kwargs.items():
            args.append(f"--{key}")
            args.append(value)

        main(args, blocking)


class DLJobBuilder(object):
    def __init__(self):
        self._dl_type = ""
        self._node_num = 0
        self._device_per_node = 0
        self._device_type = "GPU"
        self._config = {}
        self._env = {}
        self._components: Dict[
            str, Union[None, DLTrainerConfig, DLWorkloadConfig]
        ] = {}
        self._collocations: List[Set[str]] = []
        self._stream_type = DLStreamType.TASK_STREAM

    def build(self):
        """
        Build DLJob by builder's configuration.

        Returns:
            DLJob: Unified deep learning object.

        Raises:
            InvalidDLConfiguration: If validation on configration failed.
        """

        if not self.validate():
            raise InvalidDLConfiguration()

        return DLJob(
            dl_type=self._dl_type,
            stream_type=self._stream_type,
            node_num=self._node_num,
            device_per_node=self._device_per_node,
            device_type=self._device_type,
            config=self._config,
            env=self._env,
            components=self._components,
            collocations=self._collocations,
        )

    def has_elastic_training(self):
        if InternalDLWorkloadRole.ELASTIC_ROLE in list(
            self._components.keys()
        ):
            return True
        return False

    def _validate_dlrover_run_cmd(self, cmd) -> bool:
        if not cmd:
            return False
        if not cmd.startswith("dlrover-run"):
            return False
        return True

    def validate(self) -> bool:
        if not self._dl_type:
            logger.error("'dl_type' must be set.")
            return False

        if self._node_num < 1:
            logger.error("'node_num' must be greater than 0.")
            return False

        if self._device_per_node < 1:
            logger.error("'device_per_node' must be greater than 0.")
            return False

        if self._device_type != "CPU" and self._device_type != "GPU":
            logger.error("'device_type' must be 'CPU' or 'GPU'.")
            return False

        if not self._config or (
            not isinstance(self._config, dict)
            and not isinstance(self._config, DictConfig)
        ):
            logger.error("'config' must be dict type and cannot be empty.")
            return False

        # for role components
        if self._stream_type == DLStreamType.TASK_STREAM:
            if (
                InternalDLWorkloadRole.TRAINER_ROLE
                not in list(self._components.keys())
                and not self.has_elastic_training()
            ):
                logger.error(
                    "'trainer' must be set for task stream if "
                    "elastic training not involved."
                )
                return False

        for role, component in self._components.items():
            if not component:
                logger.error(f"'{role}' must be configured.")
                return False

            if component:
                if not component._module_name or not component._class_name:
                    logger.error(
                        f"{role}'s 'module_name' and 'class_name' "
                        "cannot be empty."
                    )
                    return False

                if isinstance(component, DLTrainerConfig):  # for trainer
                    pass

                if component.role_name == InternalDLWorkloadRole.ELASTIC_ROLE:
                    # for elastic-training
                    if not self._validate_dlrover_run_cmd(
                        component.others.get("run_cmd")
                    ):
                        logger.error(
                            "dlrover-run command is invalid for "
                            "elastic training."
                        )
                        return False
                elif (
                    component.role_name == InternalDLWorkloadRole.TRAINER_ROLE
                ):
                    # for common trainer
                    pass
                elif isinstance(component, DLWorkloadConfig):
                    # for general workload
                    if component.total < 1:
                        logger.error(f"{role}'s 'num' must be greater than 0.")
                        return False
                    if component.per_node < 1:
                        logger.error(
                            f"{role}'s 'per_node' must be greater than 0."
                        )
                        return False

        # for workload collocations
        if self._collocations:
            collocations_set = set()
            for collocation in self._collocations:
                process_num_sum = 0
                for role in collocation:
                    role_config = self._components[role]
                    if role_config is None:
                        logger.error(
                            "Collocation cannot be defined without "
                            f"role definition: {role}."
                        )
                        return False
                    elif isinstance(role_config, DLTrainerConfig):
                        logger.error(
                            "Trainer cannot be defined with collocation."
                        )
                        return False

                    process_num_sum += role_config.per_node

                    if role not in collocations_set:
                        collocations_set.add(role)
                    else:
                        logger.error(
                            "The same role can only be defined once "
                            "in 'collocation'."
                        )
                        return False

                # maximum of 5 roles are supported for single device affinity
                if (
                    process_num_sum != self._device_per_node
                    and process_num_sum != 2 * self._device_per_node
                    and process_num_sum != 3 * self._device_per_node
                    and process_num_sum != 4 * self._device_per_node
                    and process_num_sum != 5 * self._device_per_node
                ):
                    logger.error(
                        "The collocation is invalid due to the device "
                        "per node not satisfied the per_node "
                        "number of role for collocation."
                    )
                    return False

        return True

    def update_component(self, role_type, role_config):
        self._components[role_type] = role_config

    def SFT_type(self):
        """
        Set the deep learning type as SFT.
        """

        return self.dl_type()

    def PRE_type(self):
        """
        Set the deep learning type as PreTrain.
        """

        return self.dl_type("PRE")

    def RL_type(self):
        """
        Set the deep learning type as RL.
        """

        return self.dl_type("RL")

    def MULTIMODAL_type(self):
        """
        Set the deep learning type as MULTIMODAL.
        """

        return self.dl_type("MULTIMODAL")

    def dl_type(self, dl_type="SFT"):
        """
        Set the deep learning type.

        Args:
            dl_type (str): The type of deep learning. Default is SFT.
                Supported: SFT, PRE, RL, MULTIMODAL
        """

        self._dl_type = dl_type
        return self

    def as_task_stream(self):
        """
        Set as task stream.
        """

        self._stream_type = DLStreamType.TASK_STREAM
        return self

    def as_data_stream(self):
        """
        Set as data stream.
        """

        self._stream_type = DLStreamType.DATA_STREAM
        return self

    def node_num(self, num=1):
        """
        Set the total number of nodes.

        Args:
            num (int): The number of nodes. Default is 1.
        """

        self._node_num = num
        return self

    def device_per_node(self, num=8):
        """
        Set the device number per node.

        Args:
            num (int): The device number of single node. Default is 8.
        """

        self._device_per_node = num
        return self

    def device_type(self, device_type="GPU"):
        """
        Set the device type.

        Args:
            device_type (str, optional): The device type, support: 'CPU' or
                'GPU'. Default is 'GPU'.
        """

        self._device_type = device_type
        return self

    def config(self, config=None):
        """
        Set the training configuration.

        Args:
            config (dict): The full configuration of training in dict format.
        """

        if config is None:
            config = {}
        self._config = config
        return self

    def global_env(self, env=None):
        """
        Set the global training envs.

        Args:
            env (dict, optional): The global envs of training.
        """

        if env is None:
            env = {}
        self._env.update(env)
        return self

    class _RoleConfigurator:
        def __init__(
            self, builder, role_type: str, role_config: DLWorkloadConfig
        ):
            self.builder = builder
            self.role_type = role_type
            self.role_config = role_config

        def __exit__(self, *args):
            self.builder.update_component(self.role_type, self.role_config)
            return self.builder

    def _config_trainer_role(
        self,
        module_name,
        class_name,
        trainer_type: TrainerType = TrainerType.USER_DEFINED,
        **kwargs,
    ):
        trainer_config = DLTrainerConfig(
            trainer_type, module_name, class_name, **kwargs
        )
        self._components[InternalDLWorkloadRole.TRAINER_ROLE] = trainer_config
        return self

    def _config_workload_role(
        self, role_name, module_name, class_name, **kwargs
    ):
        role_config = DLWorkloadConfig(
            role_name, module_name, class_name, **kwargs
        )
        self._components[role_name] = role_config
        self._role_configurator = self._RoleConfigurator(
            self, role_name, role_config
        )
        return self

    def workload(self, role_name, module_name, class_name):
        """
        Setup workload.

        Args:
            role_name (str): The role name of workload.
            module_name (str): The module name of workload.
            class_name (str): The class name of workload.
        """

        return self._config_workload_role(role_name, module_name, class_name)

    def dlrover_run(
        self,
        run_cmd,
        worker_module="dlrover.python.unified.trainer.elastic_workload",
        worker_cls="ElasticWorkload",
    ):
        """
        Setup elastic agent workload(use elastic agent for elastic training,
            same with 'torchrun' case).

        Args:
            run_cmd (str): The training command.
                e.g. 'dlrover-run --xxx train.py'
        """

        # set a default trainer
        self._default_trainer()

        # set workload role
        self._config_workload_role(
            InternalDLWorkloadRole.ELASTIC_ROLE,
            worker_module,
            worker_cls,
            run_cmd=run_cmd,
        )

        # resolve total(--nnodes) and per-node(--nproc_per_node) from command
        for part in run_cmd.split():
            if part.startswith("--nnodes="):
                value = part.split("=")[1]
                if ":" in value:
                    _, max_value = value.split(":")
                    self._role_configurator.role_config._num = int(max_value)
                else:
                    self._role_configurator.role_config._num = int(value)
            elif part.startswith("--nproc_per_node="):
                self._role_configurator.role_config._per_node = int(
                    part.split("=")[1]
                )

        assert self._role_configurator.role_config.total > 1
        assert (
            self._role_configurator.role_config.per_node
            == self._device_per_node
        )

        # set the cmd into config
        self._config[InternalDLConfig.ELASTIC_RUN_CMD] = run_cmd

        # set default global env
        self._env[
            NodeEnv.DLROVER_MASTER_SERVICE_TYPE
        ] = CommunicationType.COMM_SERVICE_RAY

        return self

    def _default_trainer(self):
        return self._config_trainer_role(
            "dlrover.python.unified.trainer.trainer",
            "DefaultTrainer",
            TrainerType.ELASTIC_TRAINING,
        )

    def trainer(self, module_name, class_name):
        """
        Setup trainer for user-defined task stream.

        Args:
            module_name (str): The module name of trainer.
            class_name (str): The class name of trainer.
        """

        assert self._stream_type == DLStreamType.TASK_STREAM

        return self._config_trainer_role(module_name, class_name)

    def with_collocation(self, *roles):
        """
        Set a collocation strategy, requiring that the total number of
        collocated roles on each node(per_node) must have an even relationship
        with the number of devices on each machine(device_per_node).

        Multiple role names is required.
        For example, to colocate "actor" and "rollout", it can be expressed as
        `.with_collocation("actor", "rollout")`.

        Supported roles include: actor, rollout, reference, reward, and critic,
        with support for both uppercase and lowercase. The same role can only
        be included in one collocation.

        Args:
            roles (str): Multi role names.
        """

        roles = [role.lower() for role in roles]
        self._collocations.append(set(roles))
        return self

    def with_collocation_all(self):
        """
        Set a collocation strategy for all roles.

        Notice: can be used after role definition only
        """

        roles = set()
        for role, role_config in self._components.items():
            if role == InternalDLWorkloadRole.TRAINER_ROLE or not role_config:
                continue
            roles.add(role)
        self._collocations.append(roles)
        return self

    def total(self, num=1):
        """
        Set the total number of current role.

        Args:
            num (int): The number of current role. Default is 1.
        """

        self._role_configurator.role_config._num = num
        return self

    def per_node(self, num=1):
        """
        How many current role per node.

        Args:
            num (int): The number of current role per node.
                Default is 1.
        """

        self._role_configurator.role_config._per_node = num
        return self

    def env(self, env=None):
        """
        The envs for current role.

        Args:
            env (dict, optional): The envs of current role.
                Default is {}.
        """

        if env is None:
            env = {}
        self._role_configurator.role_config._env.update(env)
        return self

    def enable_ray_auto_visible_device(self):
        """
        Enable to let ray set visible device automatically.
        """

        self._role_configurator.role_config._env.update(
            DLWorkloadEnv.RAY_SET_VISIBLE_DEVICES_ENVS
        )
        return self

    def sub_stage(self, sub_stage=None):
        """
        The sub-stage definition for current role.

        Args:
            sub_stage (list, optional): The sub-stage definition of current
            role. Default is [].
        """

        if sub_stage is None:
            sub_stage = []
        self._role_configurator.role_config._sub_stage = sub_stage
        return self


class RLJobBuilder(DLJobBuilder):
    """
    Extension job builder for Reinforcement Learning (RL).
    """

    ROLES = ["actor", "reference", "reward", "critic", "rollout"]

    def __init__(self):
        super(RLJobBuilder, self).__init__()
        self._dl_type = DLType.RL.name

    def actor(self, module_name, class_name):
        """
        Setup actor.

        Args:
            module_name (str): The module name of actor.
            class_name (str): The class name of actor.
        """

        return self.workload("actor", module_name, class_name)

    def rollout(self, module_name, class_name):
        """
        Setup rollout.

        Args:
            module_name (str): The module name of rollout.
            class_name (str): The class name of rollout.
        """

        return self.workload("rollout", module_name, class_name)

    def reference(self, module_name, class_name):
        """
        Setup reference.

        Args:
            module_name (str): The module name of reference.
            class_name (str): The class name of reference.
        """

        return self.workload("reference", module_name, class_name)

    def reward(self, module_name, class_name):
        """
        Setup reward.

        Args:
            module_name (str): The module name of reward.
            class_name (str): The class name of reward.
        """

        return self.workload("reward", module_name, class_name)

    def critic(self, module_name, class_name):
        """
        Setup critic.

        Args:
            module_name (str): The module name of actor.
            class_name (str): The class name of actor.
        """

        return self.workload("critic", module_name, class_name)

    def validate(self) -> bool:
        if not super(RLJobBuilder, self).validate():
            return False

        if "actor" not in list(self._components.keys()):
            logger.error("'actor' must be configured.")
            return False

        for role, component in self._components.items():
            if (
                role != InternalDLWorkloadRole.TRAINER_ROLE
                and role not in RLJobBuilder.ROLES
            ):
                logger.error(
                    f"{role} is invalid for rl, supported roles "
                    f"are {RLJobBuilder.ROLES}."
                )
                return False

        return True
