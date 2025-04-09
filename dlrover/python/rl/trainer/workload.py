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
import functools
import os
import time
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.constant import RLWorkloadEnv
from dlrover.python.rl.common.enums import RLRoleType
from dlrover.python.rl.remote.call_obj import RuntimeInfo


def trainer_invocation(ret_ref=False, is_async=False, timeout=10):
    """
    Decorator for timeout controlled function using.

    Args:
        ret_ref (bool, optional): Whether return the remote object ref
            directly. Default is False.
        is_async (bool, optional): Whether invoke by ray.wait(),
            when 'get_ref=' is False. Default is False.
        timeout (int, optional): The timeout(seconds) set for ray.wait(),
            when 'get_ref=' is False. Default is 10(seconds).
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        wrapped._trainer_invocation = True
        wrapped._trainer_invocation_ret_ref = ret_ref
        wrapped._trainer_invocation_async = is_async
        wrapped._trainer_invocation_async_timeout = timeout
        return wrapped

    return decorator


class BaseWorkload(ABC):
    """
    Workload is the core computational unit for RL (Reinforcement Learning)
    training. Users need to extend the current base class and implement their
    own business logic. Workloads for different roles perform distinct
    implementations, such as the actor handling policy updates and the reward
    handling reward calculations.

    Args:
        master_handle: The actor handle of RLMaster.
        config: The configuration used for training.

        name: The of current actor.
        role: The role of current workload.
        rank: Rank(parallelism index) of current workload.
        world_size: World size(parallelism size) of current workload.
        local_rank: Local rank(parallelism index per node) of current workload.
        local_world_size: Local World size(parallelism size per node) of
            current workload.
    """

    def __init__(
        self,
        master_handle: ActorHandle,
        config: DictConfig,
    ):
        self._master_handle = master_handle
        self._config = config

        self._name = os.environ[RLWorkloadEnv.NAME]
        self._role = RLRoleType[os.environ[RLWorkloadEnv.ROLE]]
        self._rank = int(os.environ[RLWorkloadEnv.RANK])
        self._world_size = int(os.environ[RLWorkloadEnv.WORLD_SIZE])
        self._local_rank = int(os.environ[RLWorkloadEnv.LOCAL_RANK])
        self._local_world_size = int(
            os.environ[RLWorkloadEnv.LOCAL_WORLD_SIZE]
        )

        self.__create_time = int(time.time())
        self.__executor = ThreadPoolExecutor(max_workers=4)

        self.__executor.submit(self._report_master)

        logger.info(
            f"Workload {self._name} created with role: {self._role}, "
            f"rank: {self._rank}, world_size: {self._world_size}, "
            f"local_rank: {self._local_rank}, "
            f"local_world_size: {self._local_world_size}."
        )

    @property
    def master_handle(self) -> ActorHandle:
        return self._master_handle

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> RLRoleType:
        return self._role

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def local_world_size(self) -> int:
        return self._local_world_size

    @property
    def torch_master_addr(self) -> str:
        if RLWorkloadEnv.MASTER_ADDR in os.environ:
            return os.environ[RLWorkloadEnv.MASTER_ADDR]
        return ""

    @property
    def torch_master_port(self) -> int:
        if RLWorkloadEnv.MASTER_PORT in os.environ:
            return int(os.environ[RLWorkloadEnv.MASTER_PORT])
        return -1

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def create_time(self):
        return self.__create_time

    def _get_actor_id(self):
        return ray.get_runtime_context().get_actor_id()

    def is_actor_role(self):
        return self._role == RLRoleType.ACTOR

    def is_rollout_role(self):
        return self._role == RLRoleType.ROLLOUT

    def is_reward_role(self):
        return self._role == RLRoleType.REWARD

    def is_ref_role(self):
        return self._role == RLRoleType.REFERENCE

    def is_critic_role(self):
        return self._role == RLRoleType.CRITIC

    """Remote call functions start"""

    def _report_master(self):
        """
        Internal function. Do not override.
        """
        info = self.get_runtime_info()
        try:
            ray.get(self._master_handle.report.remote(info))
        except Exception as e:
            logger.error(f"Report master got unexpected error: {e}")

    def ping(self):
        """
        Internal function. Do not override.
        """
        return True

    def get_runtime_info(self) -> RuntimeInfo:
        """
        Internal function. Do not override.
        """

        hostname, host_ip = env_utils.get_hostname_and_ip()
        return RuntimeInfo(
            name=self.name,
            create_time=self.create_time,
            hostname=hostname,
            host_ip=host_ip,
        )

    def setup(self, env_dict: Dict[str, str]) -> bool:
        """
        Internal function. Do not override.
        """

        # update envs
        for key, value in env_dict.items():
            os.environ[key] = value
            logger.info(f"Setup env: {key}-{value}")

        return True

    """Remote call functions end"""
