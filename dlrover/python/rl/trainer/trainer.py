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

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.util.common_util import get_methods_by_class


class RoleGroupProxy(object):
    def __init__(
        self,
        role: RLRoleType,
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
            role: The role type.
            role_cls: The class of the target role.
            role_methods: All the methods exposed by '@trainer_invocation'.
                Format: key is the method name, value is a tuple:
                (${method}, ${is_async}, ${async_timeout})
            actor_handles: All the actor handles of current role.
        """

        self._role = role
        self._role_cls = role_cls
        self._role_methods = role_methods
        self._actor_handles = actor_handles

    def __getattr__(self, method_name: str):
        def remote_method(*args, **kwargs):
            if not self._actor_handles:
                raise RuntimeError("Actor handle is empty.")

            if method_name in self._role_methods:
                method_attr = self._role_methods[method_name]
                use_wait = method_attr[1]
                wait_timeout = method_attr[2]
            else:
                use_wait = False
                wait_timeout = 10

            refs = [
                getattr(actor, method_name).remote(*args, **kwargs)
                for actor in self._actor_handles
            ]
            if use_wait:
                ready, unready = ray.wait(refs, timeout=wait_timeout)
                return ray.get(ready)
            else:
                return ray.get(refs)

        return remote_method


class BaseTrainer(ABC):
    """
    The trainer is the core logic for RL (Reinforcement Learning) training.
    Users need to extend the current base class and implement their own
    business logic. In the trainer, users operate the workloads of different
    RL roles to complete the training process.

    Args:
        actor_handles: All the actor handles in dict format by role type.
        actor_classes: The class type in dict format by role type.
        config: The configuration used for training.
    """

    def __init__(
        self,
        actor_handles: Dict[RLRoleType, List[ActorHandle]],
        actor_classes: Dict[RLRoleType, type],
        config: DictConfig,
    ):
        self._actor_handles = actor_handles
        self._actor_classes = actor_classes
        self._config = config

        self.__role_group_proxy: Dict[str, RoleGroupProxy] = {}
        self.__init_role_group_proxy()
        logger.info(
            f"Trainer initiated with workloads: {self._get_workloads_size()}, "
            f"role actor classes: {self._actor_classes}, "
            f"config: {config}"
        )

    def _get_workloads_size(self) -> Dict[str, int]:
        return {
            role.name: len(handles)
            for role, handles in self._actor_handles.items()
        }

    def __init_role_group_proxy(self):
        for role, handles in self._actor_handles.items():
            workload_group_name = "RG_" + role.name
            cls = self._actor_classes[role]

            # get methods by class
            invocation_methods = {}
            for name, method in get_methods_by_class(cls):
                if hasattr(method, "_trainer_invocation"):
                    is_async = getattr(method, "_trainer_invocation_async")
                    async_timeout = getattr(
                        method, "_trainer_invocation_async_timeout"
                    )
                    invocation_methods[name] = (
                        method,
                        is_async,
                        async_timeout,
                    )

            role_group_proxy = RoleGroupProxy(
                role, cls, invocation_methods, handles
            )
            setattr(self, workload_group_name, role_group_proxy)
            self.__role_group_proxy[workload_group_name] = role_group_proxy

    @property
    def actor_handles(self) -> Dict[RLRoleType, List[ActorHandle]]:
        """Return all the actor handles"""
        return self._actor_handles

    @property
    def actors(self) -> List[ActorHandle]:
        """Get all actors' actor handle."""
        return self.get_actor_handles(RLRoleType.ACTOR)

    @property
    def references(self) -> List[ActorHandle]:
        """Get all references' actor handle."""
        return self.get_actor_handles(RLRoleType.REFERENCE)

    @property
    def rollouts(self) -> List[ActorHandle]:
        """Get all rollouts' actor handle."""
        return self.get_actor_handles(RLRoleType.ROLLOUT)

    @property
    def rewards(self) -> List[ActorHandle]:
        """Get all rewards' actor handle."""
        return self.get_actor_handles(RLRoleType.REWARD)

    @property
    def critics(self) -> List[ActorHandle]:
        """Get all critics' actor handle."""
        return self.get_actor_handles(RLRoleType.CRITIC)

    @property
    def config(self) -> DictConfig:
        return self._config

    def get_actor_handles(self, role_type: RLRoleType) -> List[ActorHandle]:
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
