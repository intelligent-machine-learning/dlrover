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
from typing import Dict, List, Tuple

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from dlrover.python.common.log import default_logger as logger
from dlrover.python.util.common_util import get_methods_by_class


class RoleGroupProxy(object):
    def __init__(
        self,
        role: str,
        world_size: int,
        role_cls: type,
        role_methods: Dict[str, Tuple],
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
        self._role_methods = role_methods
        self._actor_handles = actor_handles

        logger.info(
            f"Initiate role-group-proxy with role: {role}, "
            f"world_size: {world_size}"
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

            if method_name in self._role_methods:
                method_attr = self._role_methods[method_name]
                blocking = method_attr[1]
                use_wait = method_attr[2]
                wait_timeout = method_attr[3]
                target = method_attr[4]
                auto_shard = method_attr[5]
                pre_func = method_attr[6]
                post_func = method_attr[7]
            else:
                blocking = True
                use_wait = False
                wait_timeout = 10
                target = "ALL"
                auto_shard = True
                pre_func = None
                post_func = None

            logger.info(
                "Role group proxy method invocation, "
                f"method: {method_name}, "
                f"use_wait: {use_wait}, "
                f"wait_timeout: {wait_timeout}, "
                f"target: {target}, "
                f"auto_shard: {auto_shard}, "
                f"pre_func: {pre_func}, "
                f"post_func: {post_func}"
            )
            # pre function invocation
            if pre_func:
                logger.debug(
                    f"{method_name} [PRE-FUNC]: {pre_func.__name__}, "
                    f"input args({len(args)}): {[type(arg) for arg in args]}, "
                    f"input kwargs({len(kwargs)}): "
                    f"{[type(v) for v in kwargs.values()]}"
                )
                args, kwargs = pre_func(self, *args, **kwargs)
                logger.debug(
                    f"{method_name} [PRE-FUNC]: {pre_func.__name__}, "
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
            if target == "RANK0":
                logger.debug(f"{method_name} do rank0 invocation")
                refs = [
                    getattr(self._actor_handles[0], method_name).remote(
                        *args, **kwargs
                    )
                ]
            else:
                logger.debug(f"{method_name} do all invocation")
                if auto_shard and self._can_shard_invocation(*args, **kwargs):
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

            if blocking:
                if use_wait:
                    ready, unready = ray.wait(refs, timeout=wait_timeout)
                    result = ray.get(ready)
                else:
                    result = ray.get(refs)
            else:
                result = refs

            # post function invocation
            if post_func:
                logger.debug(
                    f"{method_name} [POST-FUNC]: {post_func.__name__}, "
                    f"input args({len(args)}): {[type(arg) for arg in args]}, "
                    f"input kwargs({len(kwargs)}): "
                    f"{[type(v) for v in kwargs.values()]}"
                )
                result = post_func(self, result)
                logger.debug(
                    f"{method_name} [POST-FUNC]: {post_func.__name__}, "
                    f"output args({len(args)}): "
                    f"{[type(arg) for arg in args]}, "
                    f"output kwargs({len(kwargs)}): "
                    f"{[type(v) for v in kwargs.values()]}"
                )

            return result

        return remote_method


class BaseTrainer(ABC):
    """
    The trainer is the core logic for deep learning training.
    Users need to extend the current base class and implement their own
    business logic. In the trainer, users operate the workloads of different
    DL roles to complete the training process.

    Args:
        actor_handles: All the actor handles in dict format by role type.
        actor_metas: The class meta in dict format by role type.
            Format:
                key - role
                value - tuple with: class-type, device

        config: The configuration used for training.
    """

    def __init__(
        self,
        actor_handles: Dict[str, List[ActorHandle]],
        actor_metas: Dict[str, Tuple[type, float]],
        config: DictConfig,
    ):
        self._actor_handles = actor_handles
        self._actor_metas = actor_metas
        self._config = config

        self.__role_group_proxy: Dict[str, RoleGroupProxy] = {}
        self.__init_role_group_proxy()
        logger.info(
            f"Trainer initiated with workloads: {self._get_workloads_size()}, "
            f"role actor metas: {self._actor_metas}, "
            f"config: {config}"
        )

    def _get_workloads_size(self) -> Dict[str, int]:
        return {
            role: len(handles) for role, handles in self._actor_handles.items()
        }

    def __init_role_group_proxy(self):
        for role, handles in self._actor_handles.items():
            workload_group_name = "RG_" + role
            cls = self._actor_metas[role][0]

            # get methods by class
            invocation_methods = {}
            for name, method in get_methods_by_class(cls):
                if hasattr(method, "_trainer_invocation"):
                    blocking = getattr(method, "_trainer_invocation_blocking")
                    is_async = getattr(method, "_trainer_invocation_async")
                    async_timeout = getattr(
                        method, "_trainer_invocation_async_timeout"
                    )
                    target = getattr(method, "_trainer_invocation_target")
                    auto_shard = getattr(
                        method, "_trainer_invocation_auto_shard"
                    )
                    pre_func = getattr(method, "_trainer_invocation_pre_func")
                    post_func = getattr(
                        method, "_trainer_invocation_post_func"
                    )
                    invocation_methods[name] = (
                        method,
                        blocking,
                        is_async,
                        async_timeout,
                        target,
                        auto_shard,
                        pre_func,
                        post_func,
                    )

            role_group_proxy = RoleGroupProxy(
                role,
                len(self._actor_handles[role]),
                cls,
                invocation_methods,
                handles,
            )
            setattr(self, workload_group_name, role_group_proxy)
            self.__role_group_proxy[workload_group_name] = role_group_proxy

    @property
    def actor_handles(self) -> Dict[str, List[ActorHandle]]:
        """Return all the actor handles"""
        return self._actor_handles

    @property
    def config(self) -> DictConfig:
        return self._config

    def get_workload_resource(self, role_type: str):
        """Return the workload's resource by role type."""
        if role_type in self._actor_metas:
            return self._actor_metas[role_type][1]
        return 1

    def get_actor_handles(self, role_type: str) -> List[ActorHandle]:
        """Return the actor handles by role type."""
        if role_type in self._actor_handles:
            return self._actor_handles[role_type]
        return []

    def get_role_groups(self):
        return list(self.__role_group_proxy.keys())

    def is_recoverable(self):
        """
        Indicates whether the current trainer process (or thread)
        can be directly resumed. Default is False, can override by user.

        False: Indicates that it cannot be directly resumed. If the trainer
        exits abnormally, upon restarting, a global reset will be triggered,
        meaning all workloads will be rebuilt and the trainer
        logic (init + fit) will be re-executed.

        True: Indicates that it can be directly resumed. If the trainer exits
        abnormally, upon restarting, as long as there are no other workload
        exceptions, only the trainer logic (init + fit) will be re-executed.

        """
        return False

    @abstractmethod
    def init(self):
        """
        Requires user implementation: for initializing all workers,
        such as loading the dataloader, loading the model, etc.
        """

    @abstractmethod
    def fit(self):
        """
        Requires user implementation: the core training logic.
        """


class DefaultTrainer(BaseTrainer):
    """
    Internal default trainer, such as: for elastic training case.
    """

    def init(self):
        pass

    def fit(self):
        pass
