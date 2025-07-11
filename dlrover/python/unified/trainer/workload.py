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
import functools
import os
import time
import types
from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import DLWorkloadEnv
from dlrover.python.unified.remote.call_obj import RuntimeInfo


def trainer_invocation(
    blocking=True,
    is_async=False,
    timeout=10,
    target="ALL",
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

        wrapped._trainer_invocation = True
        wrapped._trainer_invocation_blocking = blocking
        wrapped._trainer_invocation_async = is_async
        wrapped._trainer_invocation_async_timeout = timeout
        wrapped._trainer_invocation_target = target
        wrapped._trainer_invocation_auto_shard = auto_shard
        wrapped._trainer_invocation_pre_func = pre_func
        wrapped._trainer_invocation_post_func = post_func
        return wrapped

    return decorator


class BaseWorkload(ABC):
    """
    Workload is the core computational unit for deep learning training.
    Users need to extend the current base class and implement their
    own business logic. Workloads for different roles perform distinct
    implementations, such as the actor handling policy updates and the reward
    handling reward calculations.

    Args:
        master_handle: The actor handle of DLMaster.
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

        self._name = os.environ[DLWorkloadEnv.NAME]
        self._role = os.environ[DLWorkloadEnv.ROLE]
        self._rank = int(os.environ[DLWorkloadEnv.RANK])
        self._world_size = int(os.environ[DLWorkloadEnv.WORLD_SIZE])
        self._local_rank = int(os.environ[DLWorkloadEnv.LOCAL_RANK])
        self._local_world_size = int(
            os.environ[DLWorkloadEnv.LOCAL_WORLD_SIZE]
        )

        self.__create_time = int(time.time())
        self.__executor = ThreadPoolExecutor(max_workers=4)

        self.__executor.submit(self._report_runtime_info)
        if (
            ray.is_initialized()
            and ray.get_runtime_context().get_actor_id()
            and ray.get_runtime_context().was_current_actor_reconstructed
        ):
            self.__executor.submit(self._report_restarting)

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
    def role(self) -> str:
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
        if DLWorkloadEnv.MASTER_ADDR in os.environ:
            return os.environ[DLWorkloadEnv.MASTER_ADDR]
        return ""

    @property
    def torch_master_port(self) -> int:
        if DLWorkloadEnv.MASTER_PORT in os.environ:
            return int(os.environ[DLWorkloadEnv.MASTER_PORT])
        return -1

    @property
    def config(self) -> DictConfig:
        return self._config

    @property
    def create_time(self):
        return self.__create_time

    def _get_actor_id(self):
        return ray.get_runtime_context().get_actor_id()

    def get_device_collocation(self):
        return env_utils.get_env(DLWorkloadEnv.DEVICE_COLLOCATION_GROUP)

    def has_device_collocation(self):
        if (
            self.get_device_collocation()
            and self.role.name in self.get_device_collocation()
        ):
            return True
        return False

    def get_restart_info(self):
        """
        Return info for failure.
        format: (restart_time, level, reason, extra_info)
        """
        return int(time.time()), -1, "unknown", {}

    """Remote call functions start"""

    def _report_restarting(self):
        """
        Internal function. Do not override.
        Should be invoked once when actor restarts.
        """
        try:
            restart_info = self.get_restart_info()
            ray.get(
                self._master_handle.report_restarting.remote(
                    self.name,
                    restart_info[0],
                    restart_info[1],
                    restart_info[2],
                    restart_info[3],
                )
            )
        except Exception as e:
            logger.error(
                f"Report restarting to master got unexpected error: {e}"
            )
            raise e

    def _report_runtime_info(self):
        """
        Internal function. Do not override.
        """
        info = self.get_runtime_info()
        try:
            ray.get(self._master_handle.report_runtime.remote(info))
        except Exception as e:
            logger.error(
                f"Report runtime info to master got unexpected error: {e}"
            )
            raise e

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


class BaseTaskProcessingWorkload(BaseWorkload, ABC):
    """
    Basic workload abstraction for task stream.
    """

    pass


class BaseDataProcessingWorkload(BaseWorkload, ABC):
    """
    Basic workload abstraction for data stream.
    """

    pass
