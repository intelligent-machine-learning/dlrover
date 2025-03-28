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

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.enums import ModelParallelismArcType, RLRoleType
from dlrover.python.rl.remote.call_obj import RuntimeInfo


class BaseWorkload(ABC):
    def __init__(
        self,
        master_handle: ActorHandle,
        name: str,
        role: RLRoleType,
        rank: int,
        world_size: int,
        config: DictConfig,
    ):
        self._master_handle = master_handle
        self._name = name
        self._role = role
        self._rank = rank
        self._world_size = world_size
        self._config = config

        self.__create_time = int(time.time())
        self.__executor = ThreadPoolExecutor(max_workers=4)

        self.__executor.submit(self._report_master)

        logger.info(
            f"Workload {name} created with role: {role}, "
            f"rank: {rank}, world_size: {world_size}."
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
    def config(self) -> DictConfig:
        return self._config

    @property
    def create_time(self):
        return self.__create_time

    def _get_actor_id(self):
        return ray.get_runtime_context().get_actor_id()

    def _report_master(self):
        """
        Base function. Do not override.
        """
        info = self.get_runtime_info()
        try:
            ray.get(self._master_handle.report.remote(info))
        except Exception as e:
            logger.error(f"Report master got unexpected error: {e}")

    def ping(self):
        """
        Base function. Do not override.
        """
        return True

    def get_runtime_info(self) -> RuntimeInfo:
        """
        Base function. Can be extended only.
        """

        hostname, host_ip = env_utils.get_hostname_and_ip()
        return RuntimeInfo(
            name=self.name,
            create_time=self.create_time,
            hostname=hostname,
            host_ip=host_ip,
        )

    @abstractmethod
    def get_model_arc(self) -> ModelParallelismArcType:
        """
        Implement by subclasses.
        Return the 'ModelParallelismArcType' used.
        """
