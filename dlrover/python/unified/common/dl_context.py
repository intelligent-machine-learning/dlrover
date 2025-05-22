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
import traceback
from typing import Dict, List, Tuple, Union

from omegaconf import DictConfig, ListConfig

from dlrover.python.common.enums import ResourceType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.resource import Resource
from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.unified.common.constant import DLTrainerConstant
from dlrover.python.unified.common.enums import RLRoleType, TrainerType
from dlrover.python.unified.common.exception import InvalidDLConfiguration
from dlrover.python.util.common_util import get_class_by_module_and_class_name


class TrainerDesc(object):
    def __init__(
        self,
        module_class,
        trainer_type,
        node_number=0,
        device_type=DLTrainerConstant.DEVICE_TYPE_DEFAULT,
        device_per_node=DLTrainerConstant.DEVICE_PER_NODE_DEFAULT,
        torch_master_port=DLTrainerConstant.TORCH_MASTER_PORT_DEFAULT,
    ):
        """
        Description of a trainer.

        Args:
            module_class (str,str): The module and class of the trainer.
            trainer_type (str): The trainer type. Must be the type of
                'TrainerType'. Default is 'USER_DEFINED'.
            node_number (int): How many nodes.
            device_type (ResourceType, optional): Device type: 'GPU' or 'CPU'.
                Default is 'GPU'.
            device_per_node (int, optional): How many gpu(cpu) per node.
                Default is 8.
            torch_master_port (int, optional): The port used for torch
                rendezvous(for env 'MASTER_PORT'), must > 1000.
                Default is [21111, 22222, 23333, 24444, 25555].
        """
        self._module_class: Tuple[str, str] = module_class
        self._trainer_type: TrainerType = TrainerType[trainer_type]
        self._node_number: int = node_number
        self._device_type: ResourceType = ResourceType[device_type.upper()]
        self._device_per_node: int = device_per_node
        self._torch_master_port: List[int] = torch_master_port

    def __repr__(self):
        return (
            f"Trainer(class={self._module_class}, "
            f"type={self._trainer_type}, "
            f"node_number={self._node_number}, "
            f"device_type={self._device_type}, "
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
    def node_number(self):
        return self._node_number

    @property
    def device_type(self):
        return self._device_type

    @property
    def device_per_node(self):
        return self._device_per_node

    @property
    def torch_master_port(self):
        return self._torch_master_port


class WorkloadDesc(object):
    def __init__(
        self, module_class, num, resource, resource_type: ResourceType, env
    ):
        """
        Description of a workload.

        Args:
            module_class: The module and class of the workload.
            num: The number of the workload instance.
            resource: The resource of the workload instance.
            env: The env dict of the workload instance.
        """
        self._module_class: Tuple[str, str] = module_class
        self._num: int = num
        if resource:
            adjusted_resource = Resource.from_dict(resource)
        else:
            if resource_type.name == ResourceType.CPU.name:
                adjusted_resource = Resource.default_cpu()
            else:
                adjusted_resource = Resource.default_gpu()

        self._resource: Resource = adjusted_resource
        self._env: Dict[str, str] = env

    def __repr__(self):
        return (
            f"Workload(class={self._module_class}, "
            f"num={self._num}, "
            f"resource={self._resource}, "
            f"env={self._env})"
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

    @property
    def instance_env(self):
        return self._env


class WorkloadGroupDesc(object):
    def __init__(self, groups, capacity=0, unit=ResourceType.GPU):
        """
        Description of a workload group(for scheduling).

        Args:
            groups: The number of the workload instance in group.
                Format: [({${role}:${group_size}}, ${group_num},
                ${resource_unit}), (${role}, ${group_num}, ${resource_unit})].
            capacity (int, optional): The resource capacity for each group.
                Default is 0(no limit).
            unit (ResourceType, optional): The resource unit if capacity is
                specified. Default is 'GPU'.
        """
        self._groups: List[
            Tuple[Dict[str, int], int, Union[int, float]]
        ] = groups
        self._capacity: int = capacity
        self._unit: ResourceType = unit

    def __repr__(self):
        return (
            f"WorkloadGroupDesc("
            f"groups={self._groups}, "
            f"capacity={self._capacity}, "
            f"unit={self._unit})"
        )

    @classmethod
    def build(
        cls,
        groups,
        roles: Dict[str, int],
        unit_resource: Dict[str, Union[int, float]],
        capacity,
        unit,
    ):
        result: List[Tuple[Dict[str, int], int, Union[int, float]]] = []

        # for role in groups
        role_in_groups = []
        if groups:
            for group in groups:
                each_allocation = {}
                group_num = 0
                resource_unit: Union[int, float] = 1
                for key, value in group.items():
                    role = key.upper()
                    each_allocation[role] = value
                    role_in_groups.append(role)
                    if group_num == 0:
                        group_num = roles[role] / value
                    else:
                        assert group_num == roles[role] / value

                    if unit_resource[role] < 1:
                        resource_unit = unit_resource[role]

                assert int(group_num) == group_num and group_num >= 1

                result.append((each_allocation, int(group_num), resource_unit))

        # role not in groups
        for role, instance_num in roles.items():
            if role in role_in_groups:
                continue
            group_size = capacity / (unit_resource[role])
            assert int(group_size) == group_size and group_size >= 1

            group_num = instance_num / group_size
            assert int(group_num) == group_num and group_num >= 1

            result.append(
                ({role: int(group_size)}, int(group_num), unit_resource[role])
            )

        return WorkloadGroupDesc(result, capacity, unit)

    @property
    def groups(
        self,
    ) -> List[Tuple[Dict[str, int], int, Union[int, float]]]:
        return self._groups

    @property
    def capacity(self):
        return self._capacity

    @property
    def unit(self) -> ResourceType:
        return self._unit

    def split_groups_in_dict(
        self,
    ) -> Tuple[
        Dict[str, Tuple[Dict[str, int], int, Union[int, float]]],
        Dict[str, Tuple[Dict[str, int], int, Union[int, float]]],
    ]:
        device_colocate_group: Dict[
            str, Tuple[Dict[str, int], int, Union[int, float]]
        ] = {}
        other_group: Dict[
            str, Tuple[Dict[str, int], int, Union[int, float]]
        ] = {}
        for group in self._groups:
            if group[2] < 1:
                name = "COLOCATE_" + str(len(device_colocate_group))
                device_colocate_group[name] = group
            else:
                name = "COMMON_" + str(len(other_group))
                other_group[name] = group

        return device_colocate_group, other_group

    def has_device_collocate(self):
        for group in self._groups:
            if group[2] < 1:
                return True
        return False

    def cal_total_resource(self):
        sum_resource = 0
        for group in self._groups:
            group_size = sum(group[0].values())
            group_num = group[1]
            group_resource_unit = group[2]
            sum_resource += group_size * group_resource_unit * group_num

        return sum_resource

    def validate(self) -> bool:
        role_set = set()

        for group in self.groups:
            group_desc = group[0]
            group_resource_unit = group[2]

            # check duplicate role definition
            for role in list(group_desc.keys()):
                if role in role_set:
                    return False
                role_set.add(role)

            if group_resource_unit < 1:
                # the role number in group must be equal
                if not len(set(group_desc.values())) == 1:
                    return False

            # check instance num <-> resource consistency
            if self.capacity < int(
                sum(group_desc.values()) * group_resource_unit
            ):
                return False

        return True


class DLContext(PickleSerializable):
    def __init__(
        self,
        config: DictConfig,
        trainer: TrainerDesc,
        workloads: Dict[str, WorkloadDesc],
        workload_group: WorkloadGroupDesc,
        env,
    ):
        """
        Description of reinforcement learning's computing architecture.

        Args:
            config: The full configuration of rl training.
            trainer: The description for the trainer.
            workloads: A dictionary of workloads, including: actor_workload,
                rollout_workload, ref_workload, reward_workload,
                critic_workload.
            workload_group: The group definition when scheduling the
                different role workloads.
            env (dict, optional): The global env.
        """

        self._config: DictConfig = config
        self._trainer = trainer
        self._workloads = workloads
        self._workload_group = workload_group
        self._env = env

    def __repr__(self):
        return (
            f"DLContext("
            f"config:{self.config}, "
            f"env:{self.env}, "
            f"trainer:{self.trainer}, "
            f"group:{self.workload_group}, "
            f"workload:{self.workloads})"
        )

    @property
    def config(self):
        return self._config

    @property
    def env(self):
        return self._env

    @property
    def trainer(self):
        return self._trainer

    @property
    def workload_group(
        self,
    ) -> WorkloadGroupDesc:
        return self._workload_group

    @property
    def workloads(self) -> Dict[str, WorkloadDesc]:
        return self._workloads

    @classmethod
    def build_from_args(cls, args):
        conf: DictConfig = args.dl_config
        if not conf:
            raise InvalidDLConfiguration()

        try:
            config = conf.get("config")
            env = conf.get("env", {})

            # trainer
            trainer_conf = conf.get("trainer")
            trainer_desc = TrainerDesc(
                (trainer_conf.get("module"), trainer_conf.get("class")),
                trainer_conf.get("type"),
                trainer_conf.get("node_number"),
                trainer_conf.get(
                    "device_type",
                    DLTrainerConstant.DEVICE_TYPE_DEFAULT,
                ),
                trainer_conf.get(
                    "device_per_node",
                    DLTrainerConstant.DEVICE_PER_NODE_DEFAULT,
                ),
                trainer_conf.get(
                    "torch_master_port",
                    DLTrainerConstant.TORCH_MASTER_PORT_DEFAULT,
                ),
            )

            # workload
            workload_dict = {}
            wl_conf: DictConfig = conf.get("workload")
            for role_str, workload_desc_dict in wl_conf.items():
                role_type = role_str.upper()
                workload_desc = WorkloadDesc(
                    (
                        workload_desc_dict.get("module"),
                        workload_desc_dict.get("class"),
                    ),
                    workload_desc_dict.get("num"),
                    workload_desc_dict.get("resource"),
                    trainer_desc.device_type,
                    workload_desc_dict.get("env", {}),
                )
                workload_dict[role_type] = workload_desc

            # workload group
            workload_group_conf: ListConfig = conf.get("workload_group", None)
            num_for_roles = {}
            resource_for_roles = {}
            for role, workload_desc in workload_dict.items():
                if trainer_desc.device_type == ResourceType.GPU:
                    key_resource = workload_desc.instance_resource.gpu
                else:
                    key_resource = workload_desc.instance_resource.cpu
                if key_resource == 0:
                    key_resource = 1
                resource_for_roles[role] = key_resource
                num_for_roles[role] = workload_desc.instance_number

            workload_group = WorkloadGroupDesc.build(
                workload_group_conf,
                num_for_roles,
                resource_for_roles,
                trainer_desc.device_per_node,
                trainer_desc.device_type,
            )

            return cls(
                config,
                trainer_desc,
                workload_dict,
                workload_group,
                env,
            )
        except Exception as e:
            logger.error(
                f"Got invalid arguments while building RLContext: {e}",
                f"{traceback.format_exc()}",
            )
            raise InvalidDLConfiguration()

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
                if self.trainer.node_number <= 0:
                    logger.error("Node number is invalid.")
                    return False

                # device per node
                if self.trainer.device_per_node <= 0:
                    logger.error("Device per node is invalid.")
                    return False

                # torch master prot
                if len(self.trainer.torch_master_port) >= 5 and (
                    any(port < 1000 for port in self.trainer.torch_master_port)
                ):
                    logger.error("Torch master port is invalid.")
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
            if not self.workload_group.validate():
                logger.error("Workload group validation failed.")
                return False

            # instance resource consistency
            total_resource = (
                self.trainer.device_per_node * self.trainer.node_number
            )

            if total_resource < int(self.workload_group.cal_total_resource()):
                logger.error(
                    "Workload group validation failed: instance "
                    "number and resource demands > the "
                    "node number and resource."
                )
                return False

        except Exception as e:
            logger.error(
                f"Unexpected error when validate dl context: {e}, "
                f"{traceback.format_exc()}"
            )
            return False

        return True

    def has_workload_group(self) -> bool:
        return bool(self.workload_group)


class RLContext(DLContext):
    @property
    def actor_workload(self):
        if RLRoleType.ACTOR.name in self._workloads:
            return self._workloads[RLRoleType.ACTOR.name]
        return None

    @property
    def rollout_workload(self):
        if RLRoleType.ROLLOUT.name in self._workloads:
            return self._workloads[RLRoleType.ROLLOUT.name]
        return None

    @property
    def ref_workload(self):
        if RLRoleType.REFERENCE.name in self._workloads:
            return self._workloads[RLRoleType.REFERENCE.name]
        return None

    @property
    def reward_workload(self):
        if RLRoleType.REWARD.name in self._workloads:
            return self._workloads[RLRoleType.REWARD.name]
        return None

    @property
    def critic_workload(self):
        if RLRoleType.CRITIC.name in self._workloads:
            return self._workloads[RLRoleType.CRITIC.name]
        return None

    def validate(self) -> bool:
        if not super().validate():
            return False

        try:
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
        except Exception as e:
            logger.error(
                f"Unexpected error when validate rl context: {e}, "
                f"{traceback.format_exc()}"
            )
            return False

        return True
