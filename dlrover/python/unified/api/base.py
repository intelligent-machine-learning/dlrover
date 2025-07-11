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
from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Union

from omegaconf import DictConfig

from dlrover.python.common.constants import CommunicationType, NodeEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import (
    DLWorkloadEnv,
    InternalDLConfig,
    InternalDLWorkloadRole,
)
from dlrover.python.unified.common.enums import DLStreamType, TrainerType
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

    def set_num(self, num):
        self._num = num

    def set_per_node(self, per_node):
        self._per_node = per_node

    def set_env(self, env):
        self._env = env

    def set_sub_stage(self, sub_stage):
        self._sub_stage = sub_stage


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
            args.append(f"{value}")

        main(args, blocking)


class RoleBuilder(ABC):
    def __init__(self, job_builder, role, module_name, class_name):
        self._job_builder = job_builder

        self._role = role
        self._module_name = module_name
        self._class_name = class_name

        self._job_builder.add_role_builder(self)

    def __getattr__(self, attr):
        if hasattr(self._job_builder, attr):
            return getattr(self._job_builder, attr)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{attr}'"
        )

    def get_role(self):
        return self._role

    def _validate(self) -> bool:
        if not self._role or not self._module_name or not self._class_name:
            logger.error("Role or module or class not set.")
            return False
        return True

    @abstractmethod
    def _build_role(self) -> Dict[str, DLRoleConfig]:
        pass

    @abstractmethod
    def _get_per_node(self):
        return 1

    @abstractmethod
    def _get_total(self):
        return 1


class TrainerBuilder(RoleBuilder):
    def __init__(self, job_builder, trainer_type, module_name, class_name):
        super().__init__(
            job_builder,
            InternalDLWorkloadRole.TRAINER_ROLE,
            module_name,
            class_name,
        )

        self._trainer_type = trainer_type

    def _build_role(self) -> Dict[str, DLRoleConfig]:
        trainer = DLTrainerConfig(
            self._trainer_type, self._module_name, self._class_name
        )
        return {InternalDLWorkloadRole.TRAINER_ROLE: trainer}

    def _get_per_node(self):
        return 0

    def _get_total(self):
        return 1


class WorkloadBuilder(RoleBuilder):
    def __init__(self, job_builder, role, module_name, class_name):
        super().__init__(job_builder, role, module_name, class_name)

        self._num = 0
        self._per_node = 0
        self._env = {}
        self._sub_stage = []

    def total(self, num=1):
        """
        Set the total number of current role.

        Args:
            num (int): The number of current role. Default is 1.
        """

        self._num = num
        return self

    def per_node(self, num=1):
        """
        How many current role per node.

        Args:
            num (int): The number of current role per node.
                Default is 1.
        """

        self._per_node = num
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
        self._env.update(env)
        return self

    def enable_ray_auto_visible_device(self):
        """
        Enable to let ray set visible device automatically.
        """

        self._env.update(DLWorkloadEnv.RAY_SET_VISIBLE_DEVICES_ENVS)
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
        self._sub_stage = sub_stage
        return self

    def _get_per_node(self):
        return self._per_node

    def _get_total(self):
        return self._num

    def _validate(self) -> bool:
        if not super()._validate():
            return False

        if self._num < 1 or self._per_node < 1:
            logger.error(
                "Total number or per node number must be greater than 1."
            )
            return False

        return True

    def _build_role(self) -> Dict[str, DLRoleConfig]:
        workload = self._create_workload_role()

        workload.set_num(self._num)
        workload.set_per_node(self._per_node)
        workload.set_env(self._env)
        workload.set_sub_stage(self._sub_stage)

        return {self._role: workload}

    def _create_workload_role(self, **kwargs):
        return DLWorkloadConfig(
            self._role, self._module_name, self._class_name, **kwargs
        )

    def build(self):
        return self._job_builder.build()


class DLRoverRunBuilder(WorkloadBuilder):
    def __init__(
        self, parent_builder, cmd="", worker_module="", worker_cls=""
    ):
        super().__init__(
            parent_builder,
            InternalDLWorkloadRole.ELASTIC_ROLE,
            worker_module,
            worker_cls,
        )

        self._cmd = cmd
        self.__init_from_cmd()

    def __init_from_cmd(self):
        # resolve total(--nnodes) and per-node(--nproc_per_node) from command
        for part in self._cmd.split():
            if part.startswith("--nnodes="):
                value = part.split("=")[1]
                if ":" in value:
                    _, max_value = value.split(":")
                    self._num = int(max_value)
                else:
                    self._num = int(value)
            elif part.startswith("--nproc_per_node="):
                self._per_node = int(part.split("=")[1])

    def __validate_dlrover_run_cmd(self, cmd) -> bool:
        if not cmd:
            return False
        if not cmd.startswith("dlrover-run"):
            return False
        return True

    def _validate(self):
        if not self._num >= 1 or not self._per_node >= 1:
            return False
        if not self.__validate_dlrover_run_cmd(self._cmd):
            logger.error(
                "dlrover-run command is invalid for " "elastic training."
            )
            return False

        return True

    @classmethod
    def __default_trainer(cls):
        return DLTrainerConfig(
            TrainerType.ELASTIC_TRAINING,
            "dlrover.python.unified.trainer.trainer",
            "DefaultTrainer",
        )

    def _build_role(self) -> Dict[str, DLRoleConfig]:
        # build elastic workload
        elastic_workload = self._create_workload_role(
            run_cmd=self._cmd,
        )
        elastic_workload.set_num(self._num)
        elastic_workload.set_per_node(self._per_node)

        return {
            InternalDLWorkloadRole.TRAINER_ROLE: self.__default_trainer(),
            self._role: elastic_workload,
        }


class DLJobBuilder(object):
    def __init__(self):
        self._dl_type = ""
        self._node_num = 0
        self._device_per_node = 0
        self._device_type = "GPU"
        self._config = {}
        self._env = {}
        self._role_builders: Dict[str, RoleBuilder] = {}
        self._collocations: List[Set[str]] = []
        self._stream_type = DLStreamType.TASK_STREAM

    def add_role_builder(self, role_builder: RoleBuilder):
        self._role_builders[role_builder.get_role()] = role_builder

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

        # build roles
        components = {}
        for role, role_builder in self._role_builders.items():
            if role_builder:
                if not role_builder._validate():
                    raise InvalidDLConfiguration()
                for (
                    role_name,
                    role_config,
                ) in role_builder._build_role().items():
                    components[role_name] = role_config

        return DLJob(
            dl_type=self._dl_type,
            stream_type=self._stream_type,
            node_num=self._node_num,
            device_per_node=self._device_per_node,
            device_type=self._device_type,
            config=self._config,
            env=self._env,
            components=components,
            collocations=self._collocations,
        )

    def has_elastic_training(self):
        if InternalDLWorkloadRole.ELASTIC_ROLE in list(
            self._role_builders.keys()
        ):
            return True
        return False

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
                not in list(self._role_builders.keys())
                and not self.has_elastic_training()
            ):
                logger.error(
                    "'trainer' must be set for task stream if "
                    "elastic training not involved."
                )
                return False

        # for workload collocations
        if self._collocations:
            collocations_set = set()
            for collocation in self._collocations:
                process_num_sum = 0
                for role in collocation:
                    role_builder = self._role_builders[role]
                    if role_builder is None:
                        logger.error(
                            "Collocation cannot be defined without "
                            f"role definition: {role}."
                        )
                        return False
                    elif isinstance(role_builder, TrainerBuilder):
                        logger.error(
                            "Trainer cannot be defined with collocation."
                        )
                        return False

                    process_num_sum += role_builder._get_per_node()

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

    def workload(self, role_name, module_name, class_name):
        """
        Setup workload.

        Args:
            role_name (str): The role name of workload.
            module_name (str): The module name of workload.
            class_name (str): The class name of workload.
        """

        return WorkloadBuilder(self, role_name, module_name, class_name)

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

        dlrover_run_builder = DLRoverRunBuilder(
            self, run_cmd, worker_module, worker_cls
        )

        # set the cmd into config
        self._config[InternalDLConfig.ELASTIC_RUN_CMD] = run_cmd

        # set default global env
        self._env[
            NodeEnv.DLROVER_MASTER_SERVICE_TYPE
        ] = CommunicationType.COMM_SERVICE_RAY

        return dlrover_run_builder

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
        for role in list(self._role_builders.keys()):
            if role == InternalDLWorkloadRole.TRAINER_ROLE:
                continue
            roles.add(role)
        self._collocations.append(roles)
        return self
