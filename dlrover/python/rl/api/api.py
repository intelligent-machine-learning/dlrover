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

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.exception import InvalidRLConfiguration
from dlrover.python.rl.driver.main import main


class RLRoleConfig(object):
    def __init__(self, module_name, class_name, **kwargs):
        self._module_name = module_name
        self._class_name = class_name
        self._kwargs = kwargs

        self._num = 1
        self._per_node = 1
        self._env = {}
        self._sub_stage = []

    @property
    def module_class(self) -> Tuple[str, str]:
        return self._module_name, self._class_name

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

    @property
    def others(self):
        return self._kwargs


class RLJob(object):
    def __init__(
        self,
        node_num,
        device_per_node,
        device_type,
        config,
        env,
        components,
        collocations,
    ):
        self._node_num = node_num
        self._device_per_node = device_per_node
        self._device_type = device_type
        self._config = config
        self._env = env
        self._components: Dict[str, Union[None, RLRoleConfig]] = components
        self._collocations: List[Set[str]] = collocations

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
        return self._components["trainer"]

    @property
    def actor(self):
        return self._components["actor"]

    @property
    def rollout(self):
        return self._components["rollout"]

    @property
    def reference(self):
        return self._components["reference"]

    @property
    def reward(self):
        return self._components["reward"]

    @property
    def critic(self):
        return self._components["critic"]

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

        # default
        return {self.device_type: 1}

    def _to_rl_config(self):
        config_result = {}

        # general
        config_result["algorithm_type"] = "USER_DEFINED"
        config_result["config"] = self.config
        config_result["env"] = self.env

        # trainer
        trainer_dict = {
            "type": self.trainer.others["trainer_type"],
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
            if role == "trainer":
                continue
            if role_config:
                role_dict = {
                    "module": role_config.module_class[0],
                    "class": role_config.module_class[1],
                    "num": role_config.total,
                    "resource": self.__cal_role_resource(role),
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
        master_cpu=4,
        master_memory=8192,
        job_max_restart=10,
        **kwargs,
    ):
        """
        Submit the current rl job.

        Args:
            job_name (str): The name of the job.
            master_cpu (int, optional): The number of CPU cores to use.
                Defaults to 4.
            master_memory (int, optional): The number of memory cores to use.
                Unit: mb. Defaults to 8192.
            job_max_restart (int, optional): The maximum number of restarts.
                Defaults to 10.

            Other arguments please refer to: 'dlrover.rl.common.args'
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
            "--rl_config",
            f"{self._to_rl_config()}",
        ]

        for key, value in kwargs.items():
            args.append(f"--{key}")
            args.append(value)

        main(args)


class RLJobBuilder(object):

    ROLES = ["actor", "reference", "reward", "critic", "rollout"]

    def __init__(self):
        self._node_num = 0
        self._device_per_node = 0
        self._device_type = "GPU"
        self._config = {}
        self._env = {}
        self._components: Dict[str, Union[None, RLRoleConfig]] = {
            "trainer": None,
            "actor": None,
            "rollout": None,
            "reference": None,
            "reward": None,
            "critic": None,
        }
        self._collocations: List[Set[str]] = []

    def build(self):
        """
        Build RLJob by builder's configuration.

        Returns:
            RLJob: RLJob object.

        Raises:
            InvalidRLConfiguration: If validation on configration failed.
        """

        if not self._validate():
            raise InvalidRLConfiguration()

        return RLJob(
            node_num=self._node_num,
            device_per_node=self._device_per_node,
            device_type=self._device_type,
            config=self._config,
            env=self._env,
            components=self._components,
            collocations=self._collocations,
        )

    def _validate(self) -> bool:
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

        if not isinstance(self._env, dict):
            logger.error("'env' must be dict type.")
            return False

        # for role components
        for role, component in self._components.items():
            if role == "trainer":
                if not component:
                    logger.error("'trainer' must be configured.")
                    return False
            if role == "actor":
                if not component:
                    logger.error("'actor' must be configured.")
                    return False

            if component:
                if not component._module_name or not component._class_name:
                    logger.error(
                        f"{role}'s 'module_name' and 'class_name' "
                        "cannot be empty."
                    )
                    return False
                if component._num < 1:
                    logger.error(f"{role}'s 'num' must be greater than 0.")
                    return False
                if component._per_node < 1:
                    logger.error(
                        f"{role}'s 'per_node' must be greater than 0."
                    )
                    return False
                if not isinstance(component._env, dict):
                    logger.error(f"{role}'s 'env' must be dict type.")
                    return False

        # for role collocations
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
                    process_num_sum += role_config.per_node

                    if role not in RLJobBuilder.ROLES:
                        logger.error(
                            f"{role} is invalid, supported roles "
                            f"are {RLJobBuilder.ROLES}."
                        )
                        return False
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
        self._env = env
        return self

    class _RoleConfigurator:
        def __init__(self, builder, role_type: str, role_config: RLRoleConfig):
            self.builder = builder
            self.role_type = role_type
            self.role_config = role_config

        def __exit__(self, *args):
            self.builder.update_component(self.role_type, self.role_config)
            return self.builder

    def _config_role(self, role, module_name, class_name, **kwargs):
        role_config = RLRoleConfig(module_name, class_name, **kwargs)
        self._components[role] = role_config
        self._role_configurator = self._RoleConfigurator(
            self, role, role_config
        )
        return self

    def trainer(self, module_name, class_name):
        """
        Setup trainer.

        Args:
            module_name (str): The module name of trainer.
            class_name (str): The class name of trainer.
        """

        return self._config_role(
            "trainer", module_name, class_name, trainer_type="USER_DEFINED"
        )

    def actor(self, module_name, class_name):
        """
        Setup actor.

        Args:
            module_name (str): The module name of actor.
            class_name (str): The class name of actor.
        """

        return self._config_role("actor", module_name, class_name)

    def rollout(self, module_name, class_name):
        """
        Setup rollout.

        Args:
            module_name (str): The module name of rollout.
            class_name (str): The class name of rollout.
        """

        return self._config_role("rollout", module_name, class_name)

    def reference(self, module_name, class_name):
        """
        Setup reference.

        Args:
            module_name (str): The module name of reference.
            class_name (str): The class name of reference.
        """

        return self._config_role("reference", module_name, class_name)

    def reward(self, module_name, class_name):
        """
        Setup reward.

        Args:
            module_name (str): The module name of reward.
            class_name (str): The class name of reward.
        """

        return self._config_role("reward", module_name, class_name)

    def critic(self, module_name, class_name):
        """
        Setup critic.

        Args:
            module_name (str): The module name of actor.
            class_name (str): The class name of actor.
        """

        return self._config_role("critic", module_name, class_name)

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
            if role == "trainer" or not role_config:
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
        self._role_configurator.role_config._env = env
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
