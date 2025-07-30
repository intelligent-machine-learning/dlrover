#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import time
import types
from abc import ABC
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from threading import Thread
from typing import Callable, Dict, List, Literal, Optional

import ray
from ray.actor import ActorHandle

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.backend.rl import remote_call
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.common.workload_base import (
    ActorBase,
    ActorInfo,
    WorkerStage,
)
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.actor_helper import get_actor_with_cache
from dlrover.python.unified.util.actor_proxy import (
    invoke_actor_t,
    invoke_actors_t,
)
from dlrover.python.util.common_util import (
    get_class_by_module_and_class_name,
    get_methods_by_class,
)


def trainer_invocation(
    blocking=True,
    is_async=False,
    timeout=10,
    target: Literal["ALL", "RANK0"] = "ALL",
    auto_shard=True,
    pre_func=None,
    post_func=None,
):
    """
    Decorator for timeout controlled function using.

    Args:
        blocking (bool, optional): Whether block until the remote result
            return. Default is True.
        is_async (bool, optional): Whether invoke by ray.wait(),
            when 'get_ref=' is False. Default is False.
        timeout (int, optional): The timeout(seconds) set for ray.wait(),
            when 'get_ref=' is False. Default is 10(seconds).
        target (str, optional): The remote invocation target.
            Support:
                ALL: All the remote actors should invoke.
                RANK0: Only the 1st actor should invoke.
            Default is 'ALL'.
        auto_shard (bool, optional): Whether enable sharding invocation when
            the length of the input parameter matches the number of target
            workloads. Default is True.
            i.e.
            split the n pieces of data, distribute them to n workloads,
            and have each workload process one part.
        pre_func (function, optional): The function will be invoked before the
            remote function.
        post_func (function, optional): The function will be invoked after the
            remote function.
    """

    assert timeout > 0
    assert target in ["ALL", "RANK0"]
    if pre_func:
        assert isinstance(pre_func, types.FunctionType)
    if post_func:
        assert isinstance(post_func, types.FunctionType)

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(
            wrapped,
            "_trainer_invocation",
            MethodInvocationMeta(
                blocking=blocking,
                is_async=is_async,
                timeout=timeout,
                target=target,
                auto_shard=auto_shard,
                pre_func=pre_func,
                post_func=post_func,
            ),
        )
        return wrapped

    return decorator


@dataclass
class MethodInvocationMeta:
    blocking: bool = True
    is_async: bool = False
    timeout: float = 10.0
    target: Literal["ALL", "RANK0"] = "ALL"
    auto_shard: bool = True
    pre_func: Optional[Callable] = None
    post_func: Optional[Callable] = None


class RoleGroupProxy(object):
    def __init__(
        self,
        role: str,
        world_size: int,
        role_cls: type,
        actor_handles: List[ActorHandle],
    ):
        """
        The role group proxy is designed to provide method call delegation for
        workloads of different roles. Through this proxy, users can directly
        call a specific method of all actor roles' actors, such as RG_ACTOR.
        Alternatively, users can bypass the proxy and directly use the
        actor_handle to perform remote method calls on any actor.

        Args:
            role: The role name.
            world_size: The world size of current role group.
            role_cls: The class of the target role.
            role_methods: All the methods exposed by '@trainer_invocation'.
                Format: key is the method name, value is a tuple:
                (${method}, ${is_async}, ${async_timeout})
            actor_handles: All the actor handles of current role.
        """

        self._role = role
        self._world_size = world_size
        self._role_cls = role_cls
        self._actor_handles = actor_handles

        # get methods by class
        invocation_methods: Dict[str, MethodInvocationMeta] = {}
        for name, method in get_methods_by_class(role_cls):
            if hasattr(method, "_trainer_invocation"):
                attr = getattr(method, "_trainer_invocation")
                assert isinstance(attr, MethodInvocationMeta)
                invocation_methods[name] = attr
        self._role_methods = invocation_methods

        logger.info(
            f"Initiate role-group-proxy with role: {role}, world_size: {world_size}"
        )

    @property
    def role(self):
        return self._role

    @property
    def world_size(self):
        return self._world_size

    def _can_shard_invocation(self, *args, **kwargs):
        if (
            all(isinstance(arg, list) for arg in args)
            and all(isinstance(kwarg, list) for kwarg in kwargs.values())
            and all(len(arg) == self.world_size for arg in args)
            and all(len(kwarg) == self.world_size for kwarg in kwargs.values())
        ):
            return True

        return False

    def __getattr__(self, method_name: str):
        def remote_method(*args, **kwargs):
            if not self._actor_handles:
                raise RuntimeError("Actor handle is empty.")

            attr = self._role_methods.get(method_name, MethodInvocationMeta())

            logger.info(
                f"Role group proxy method invocation, "
                f"method: {method_name}, {attr}"
            )
            # pre function invocation
            if attr.pre_func:
                logger.debug(
                    f"{method_name} [PRE-FUNC]: {attr.pre_func.__name__}, "
                    f"input args({len(args)}): {[type(arg) for arg in args]}, "
                    f"input kwargs({len(kwargs)}): "
                    f"{[type(v) for v in kwargs.values()]}"
                )
                args, kwargs = attr.pre_func(self, *args, **kwargs)
                logger.debug(
                    f"{method_name} [PRE-FUNC]: {attr.pre_func.__name__}, "
                    f"output args({len(args)}): "
                    f"{[type(arg) for arg in args]}, "
                    f"output kwargs({len(kwargs)}): "
                    f"{[type(v) for v in kwargs.values()]}"
                )

            # target function invocation
            logger.debug(
                f"{method_name}, "
                f"input args({len(args)}): {[type(arg) for arg in args]}, "
                f"input kwargs({len(kwargs)}): "
                f"{[type(v) for v in kwargs.values()]}"
            )
            if attr.target == "RANK0":
                logger.debug(f"{method_name} do rank0 invocation")
                refs = [
                    getattr(self._actor_handles[0], method_name).remote(
                        *args, **kwargs
                    )
                ]
            else:
                logger.debug(f"{method_name} do all invocation")
                if attr.auto_shard and self._can_shard_invocation(
                    *args, **kwargs
                ):
                    logger.debug(f"{method_name} do sharding invocation")
                    result = []
                    for i in range(self.world_size):
                        shard_args = tuple(arg[i] for arg in args)
                        shard_kwargs = {k: v[i] for k, v in kwargs.items()}
                        result.append(
                            getattr(
                                self._actor_handles[i], method_name
                            ).remote(*shard_args, **shard_kwargs)
                        )
                    refs = result
                else:
                    logger.debug(f"{method_name} do non-sharding invocation")
                    refs = [
                        getattr(actor, method_name).remote(*args, **kwargs)
                        for actor in self._actor_handles
                    ]

            if attr.blocking:
                if not attr.is_async:
                    ready, unready = ray.wait(refs, timeout=attr.timeout)
                    result = ray.get(ready)
                else:
                    result = ray.get(refs)
            else:
                result = refs

            # post function invocation
            if attr.post_func:
                logger.debug(
                    f"{method_name} [POST-FUNC]: {attr.post_func.__name__}, "
                    f"input args({len(args)}): {[type(arg) for arg in args]}, "
                    f"input kwargs({len(kwargs)}): "
                    f"{[type(v) for v in kwargs.values()]}"
                )
                result = attr.post_func(self, result)
                logger.debug(
                    f"{method_name} [POST-FUNC]: {attr.post_func.__name__}, "
                    f"output args({len(args)}): "
                    f"{[type(arg) for arg in args]}, "
                    f"output kwargs({len(kwargs)}): "
                    f"{[type(v) for v in kwargs.values()]}"
                )

            return result

        return remote_method


class BaseRLTrainer(ActorBase, ABC):
    """
    The trainer is the core logic for rl training.
    Users need to extend the current base class and implement their own
    business logic. In the trainer, users operate the workloads of different
    DL roles to complete the training process.

    Args:
        actor_handles: All the actor handles in dict format by role type.
        actor_metas: The class meta in dict format by role type.
            Format:
                key - role
                value - tuple with: class-type, device
    """

    def _setup(self):
        all_workers = {}
        for role in RLRoleType:
            if role == RLRoleType.TRAINER:
                continue
            try:
                workers = PrimeMasterApi.get_workers_by_role(role.name)
            except ValueError:
                continue
            all_workers[role] = workers

        self._workers: Dict[str, List[ActorInfo]] = all_workers
        worker_size = {
            role: len(workers) for role, workers in all_workers.items()
        }
        logger.info(
            f"Trainer initiated with workloads: {worker_size}, config: {self.config}"
        )
        self._update_stage_force(WorkerStage.READY, WorkerStage.INIT)

    def start(self):
        # Initialize the elastic client here
        logger.info("Start job execution.")
        self._setup_workloads()

        self._start_execution()

    def _setup_workloads(self):
        start = time.time()

        # setup trainer itself
        self.__init_role_group_proxy()

        # setup other workloads
        for role, group in self._workers.items():
            logger.info(f"Setting up workload: {role}...")
            master_addr = invoke_actor_t(
                remote_call.get_master_addr, group[0].name
            ).wait()

            envs = {
                DLWorkloadEnv.MASTER_ADDR: master_addr[0],
                DLWorkloadEnv.MASTER_PORT: str(master_addr[1]),
            }

            res = invoke_actors_t(
                remote_call.setup_rl_workload, group, env_dict=envs
            ).wait()
            res.raise_for_errors()

        elapsed = time.time() - start
        logger.info(
            f"Finish setup all rl workloads, cost: {elapsed / 1000:.2f}ms"
        )

    def __init_role_group_proxy(self):
        for role, workers in self._workers.items():
            workload_group_name = "RG_" + role
            assert workers[0].spec.backend == "custom"
            clz = get_class_by_module_and_class_name(
                workers[0].spec.module_name,
                workers[0].spec.class_name,
            )
            assert isinstance(clz, type)

            role_group_proxy = RoleGroupProxy(
                role,
                len(workers),
                clz,
                self.get_actor_handles(role),
            )
            setattr(self, workload_group_name, role_group_proxy)

    def _start_execution(self):
        self._update_stage_force(WorkerStage.RUNNING, WorkerStage.READY)

        @contextmanager
        def wrap_run():
            try:
                yield
                self._update_stage_force(
                    WorkerStage.FINISHED, WorkerStage.RUNNING
                )

                invoke_actors_t(
                    remote_call.update_rl_workload_stage,
                    list(chain(*self._workers.values())),
                    WorkerStage.FINISHED,
                ).wait()

            except Exception:
                logger.error(
                    "Unexpected error occurred while running rl trainer",
                    exc_info=True,
                )
                self._update_stage_force(
                    WorkerStage.FAILED, WorkerStage.RUNNING
                )

                invoke_actors_t(
                    remote_call.update_rl_workload_stage,
                    list(chain(*self._workers.values())),
                    WorkerStage.FAILED,
                ).wait()

        Thread(
            target=wrap_run()(self.trainer_run),
            daemon=True,
        ).start()

    def trainer_run(self):
        pass

    @property
    def config(self):
        return self.job_info.user_config

    @property
    def actors(self) -> List[ActorHandle]:
        """Get all actors' actor handle."""
        return self.get_actor_handles(RLRoleType.ACTOR.name)

    @property
    def references(self) -> List[ActorHandle]:
        """Get all references' actor handle."""
        return self.get_actor_handles(RLRoleType.REFERENCE.name)

    @property
    def rollouts(self) -> List[ActorHandle]:
        """Get all rollouts' actor handle."""
        return self.get_actor_handles(RLRoleType.ROLLOUT.name)

    @property
    def rewards(self) -> List[ActorHandle]:
        """Get all rewards' actor handle."""
        return self.get_actor_handles(RLRoleType.REWARD.name)

    @property
    def critics(self) -> List[ActorHandle]:
        """Get all critics' actor handle."""
        return self.get_actor_handles(RLRoleType.CRITIC.name)

    def get_actor_handles(self, role_type: str) -> List[ActorHandle]:
        """Return the actor handles by role type."""
        if role_type in self._workers:
            return [
                get_actor_with_cache(it.name)
                for it in self._workers[role_type]
            ]
        return []
