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
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    cast,
)

from pydantic import Field, model_validator

from dlrover.python.unified.common.constant import (
    DLWorkloadEnv,
    InternalDLWorkloadRole,
)
from dlrover.python.unified.common.enums import DLStreamType
from dlrover.python.unified.common.workload_desc import (
    ElasticWorkloadDesc,
    ResourceDesc,
    SimpleWorkloadDesc,
    WorkloadDesc,
)
from dlrover.python.unified.controller.config import (
    DLConfig,
    JobConfig,
)
from dlrover.python.unified.driver.main import submit

# Note: Builder don't do validation, let DLJob validate when build().


class DLJob(DLConfig):
    stream_type: DLStreamType = DLStreamType.TASK_STREAM
    collocations: List[Set[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def apply_collocations(self):
        # apply collocations
        workloads = self.workloads
        for i, collocation in enumerate(self.collocations):
            name = f"collocation_{i}"
            resource = ResourceDesc(accelerator=round(1 / len(collocation), 2))
            for role in collocation:
                if role not in workloads:
                    raise ValueError(
                        f"Role '{role}' is not defined in workloads, but is included in collocation."
                    )
                if (
                    workloads[role].group is not None
                    and workloads[role].group != name
                ):
                    raise ValueError(
                        f"Role {role} has already been assigned to a group: "
                        f"{workloads[role].group}. Cannot be included in collocation."
                    )
                workloads[role].group = name
                workloads[role].resource = resource.model_copy()
        return self

    # REVIEW - Should keep these aliases for backward compatibility?
    @property
    def node_num(self):
        return self.node_number

    @property
    def device_type(self):
        return self.accelerator_type

    @property
    def config(self):
        return self.user_config

    @property
    def env(self):
        return self.global_envs

    def get_workload(self, role):
        return self.workloads[role]

    def submit(
        self,
        job_name,
        blocking=True,
        master_cpu=4,
        master_memory=8192,
        job_max_restart=10,
        **kwargs,
    ) -> int:
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
        Returns:
            int: The exit code of the job. 0 means success, other means
                failure.
        """

        config = JobConfig(
            job_name=job_name,
            master_cpu=master_cpu,
            master_mem=master_memory,
            job_max_restart=job_max_restart,
            dl_config=self,
            **kwargs,
        )

        return submit(config, blocking)


T = TypeVar("T", bound="DLJobBuilder", covariant=True)


class RoleBuilder(ABC, Generic[T]):
    def __init__(self, parent: T, role: str, entrypoint: str):
        self.role = role
        self.entrypoint = entrypoint
        self._parent = parent

        self._num = 1
        self._per_group = 1
        self._env: Dict[str, str] = {}
        self._resource: Dict[str, Union[int, float]] = {}
        self._sub_stage: List[str] = []

    def total(self, num=1):
        """
        Set the total number of current role.

        Args:
            num (int): The number of current role. Default is 1.
        """

        self._num = num
        return self

    def per_group(self, num=1):
        """
        How many current role per group.

        Args:
            num (int): The number of current role per group.
                Default is 1.
        """

        self._per_group = num
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

    def resource(self, cpu=0, memory=0, disk=0, accelerator=0, **kwargs):
        """
        The resource for current role.

        Args:
            cpu (int, optional): The number of CPU cores to use. Defaults to 0.
            memory (int, optional): The size of memory to use. Defaults to 0. Unit: mb.
            accelerator (int, optional): The number of accelerator cores to use. Defaults to 0.
        """

        self._resource.update(
            {
                "cpu": cpu,
                "memory": memory,
                "disk": disk,
                "accelerator": accelerator,
            }
        )
        self._resource.update(kwargs)
        return self

    def disable_ray_auto_visible_device(self):
        """
        Disable to let ray set visible device automatically.
        """

        self._env.update(DLWorkloadEnv.RAY_NOSET_VISIBLE_DEVICES_ENVS)
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

    @abstractmethod
    def _build_role(self) -> Dict[str, WorkloadDesc]: ...

    def end(self) -> T:
        """Return the parent builder instance.
        It not necessary, but recommended to keep type hint working.
        """
        return self._parent

    def __getattr__(self, name: str) -> Any:
        if hasattr(self._job_builder, name):
            return getattr(self._job_builder, name)
        return super().__getattribute__(name)


class ElasticTrainBuilder(RoleBuilder[T]):
    """Builder for elastic training workload."""

    def nnodes(self, num: int):
        self._num = num * self._per_group
        return self

    def nproc_per_node(self, num: int):
        nnodes = self._num // self._per_group
        self._per_group = num
        self._num = nnodes * num
        return self

    def _build_role(self) -> Dict[str, WorkloadDesc]:
        return {
            self.role: ElasticWorkloadDesc(
                entry_point=self.entrypoint,
                envs=self._env,
                resource=ResourceDesc.get_or_default(self._resource),
                total=self._num,
                per_group=self._per_group,
            ),
        }


class SimpleWorkloadBuilder(RoleBuilder[T]):
    """Builder for simple workload."""

    def _build_role(self) -> Dict[str, WorkloadDesc]:
        return {
            self.role: SimpleWorkloadDesc(
                entry_point=self.entrypoint,
                envs=self._env,
                resource=ResourceDesc.get_or_default(self._resource),
                total=self._num,
                per_group=self._per_group,
            ),
        }


class DLJobBuilder(object):
    def __init__(self):
        # Dummy object to hold parameters, use default if not assigned.
        self._params = cast(DLJob, SimpleNamespace())

        self._env = {}
        self._role_builders: Dict[str, RoleBuilder] = {}
        self._collocations: List[Set[str]] = []
        self._last_role: Optional[str] = None

    def add_role_builder(self, role: str, role_builder: RoleBuilder):
        if role in self._role_builders:
            raise ValueError(f"Role '{role}' is already defined.")
        self._role_builders[role] = role_builder

    def build(self):
        """
        Build DLJob by builder's configuration.

        Returns:
            DLJob: Unified deep learning object.

        Raises:
            pydantic.ValidationError: If validation on configuration failed.
        """

        # build roles
        workloads: Dict[str, WorkloadDesc] = {}
        for role, role_builder in self._role_builders.items():
            if role_builder:
                workloads.update(role_builder._build_role())

        return DLJob(
            global_envs=self._env,
            workloads=workloads,
            collocations=self._collocations,
            **self._params.__dict__,
        )

    def as_task_stream(self):
        """
        Set as task stream.
        """

        self._params.stream_type = DLStreamType.TASK_STREAM
        return self

    def as_data_stream(self):
        """
        Set as data stream.
        """

        self._params.stream_type = DLStreamType.DATA_STREAM
        return self

    def node_num(self, num=1):
        """
        Set the total number of nodes.

        Args:
            num (int): The number of nodes. Default is 1.
        """
        self._params.node_number = num
        return self

    def device_per_node(self, num=8):
        """
        Set the device number per node.

        Args:
            num (int): The device number of single node. Default is 8.
        """
        self._params.device_per_node = num
        return self

    def device_type(self, device_type="GPU"):
        """
        Set the device type.

        Args:
            device_type (str, optional): The device type, support: 'CPU' or
                'GPU'. Default is 'GPU'.
        """
        # let pydantic validate later
        self._params.accelerator_type = cast(Any, device_type)
        return self

    def config(self, config=None):
        """
        Set the training configuration.

        Args:
            config (dict): The full configuration of training in dict format.
        """

        if config is None:
            config = {}
        self._params.user_config = config
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

    def role(self, role: str):
        """
        Set the last role for next workload definition.

        Args:
            role (str): The role name of next workload.
        """

        self._last_role = role
        return self

    def run(self, entrypoint: str):
        """
        Setup simple workload.

        Args:
            role (str): The role name of workload.
            entrypoint (str): The entry point of workload.
        """
        if self._last_role is None:
            raise ValueError("Role must be set before calling run().")

        role = self._last_role
        builder = SimpleWorkloadBuilder(self, role, entrypoint)
        self.add_role_builder(builder.role, builder)
        return builder

    def train(self, entrypoint: str):
        """
        Setup elastic agent workload(use elastic agent for elastic training,
            same with 'torchrun' case).

        Args:
            entrypoint (str): The training entrypoint.
                e.g. 'xxx.module.function'
        """
        role = self._last_role or InternalDLWorkloadRole.ELASTIC_ROLE
        builder = ElasticTrainBuilder(self, role, entrypoint)
        self.add_role_builder(builder.role, builder)
        return builder

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

        roles = [role.upper() for role in roles]
        self._collocations.append(set(roles))
        return self

    def with_collocation_all(self, *exclude_roles):
        """
        Set a collocation strategy for all roles.

        Notice: can be used after role definition only
        """

        roles = set()
        for role in list(self._role_builders.keys()):
            if role in exclude_roles:
                continue
            roles.add(role)
        self._collocations.append(roles)
        return self
