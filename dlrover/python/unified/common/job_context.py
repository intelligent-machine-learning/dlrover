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
import threading
from dataclasses import dataclass
from typing import Dict, List, Union

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.common.singleton import Singleton
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.dl_context import DLContext
from dlrover.python.unified.common.enums import JobStage, RLRoleType
from dlrover.python.unified.master.graph import DLExecutionGraph


@dataclass
class RestartInfo(object):

    restart_time: int = 0
    with_failover: bool = True
    reason: str = ""


class JobContext(Singleton, PickleSerializable):
    """
    JobContext includes all the key runtime information.
    """

    def __init__(self):
        self._job_config = None
        self._dl_context = None
        self._execution_graph = None

        self._job_stage = JobStage.INIT

        # will retry(invoke 'fit') trainer only(without restart other
        # workloads) if this param is set to True
        self._is_trainer_recoverable = False

        # consistent history restart info, format: type: List[RestartInfo]
        self._restart_info: Dict[str, List[RestartInfo]] = {
            "JOB": [],
            "MASTER": [],
            "TRAINER": [],
        }

        # exclude fields for serialization
        self._locker = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_locker" in state:
            del state["_locker"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._locker = threading.Lock()

    def init(self, job_config: JobConfig, dl_context: DLContext):
        self._job_config = job_config
        self._dl_context = dl_context

    @property
    def job_config(self) -> JobConfig:
        return self._job_config

    @property
    def dl_context(self) -> DLContext:
        return self._dl_context

    @property
    def execution_graph(self) -> DLExecutionGraph:
        return self._execution_graph

    def set_execution_graph(self, execution_graph: DLExecutionGraph):
        self._execution_graph = execution_graph

    def set_trainer_recoverable(self):
        logger.info("Set trainer as recoverable.")
        self._is_trainer_recoverable = True

    def is_trainer_recoverable(self):
        return self._is_trainer_recoverable

    def add_job_restart(self, restart_time, with_failover):
        self.add_restart_info(
            "JOB",
            RestartInfo(
                restart_time=restart_time, with_failover=with_failover
            ),
        )

    def add_trainer_restart_info(self, restart_info: RestartInfo):
        self._restart_info["TRAINER"].append(restart_info)

    def add_master_restart_info(self, restart_info: RestartInfo):
        self._restart_info["MASTER"].append(restart_info)

    def add_restart_info(self, restart_type: str, restart_info: RestartInfo):
        self._restart_info.setdefault(restart_type, []).append(restart_info)
        logger.info(f"Current job restart info: {self._restart_info}")

    def get_job_restart_info(self):
        return self.get_restart_info("JOB")

    def get_master_restart_info(self):
        return self.get_restart_info("MASTER")

    def get_trainer_restart_info(self):
        return self.get_restart_info("TRAINER")

    def get_workload_restart_info(self, role: Union[str, RLRoleType]):
        if isinstance(role, RLRoleType):
            role_name = role.name
        else:
            role_name = role
        return self.get_restart_info(role_name)

    def get_restart_info(self, restart_type: str):
        if restart_type not in self._restart_info:
            self._restart_info[restart_type] = []
        return self._restart_info[restart_type]

    def is_in_failover(self) -> bool:
        return self._job_stage == JobStage.FAILOVER

    def set_in_failover_stage(self):
        self.set_job_stage(JobStage.FAILOVER)

    def set_job_stage(self, job_stage: JobStage):
        self._job_stage = job_stage
        logger.info(f"Job stage updated: {job_stage}")

    def get_job_stage(self) -> JobStage:
        return self._job_stage


def get_job_context() -> JobContext:
    job_context = JobContext.singleton_instance()
    return job_context
