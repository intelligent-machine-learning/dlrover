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
import threading

from rl.master.graph import RLExecutionGraph

from dlrover.python.common.serialize import PickleSerializable
from dlrover.python.common.singleton import Singleton
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.enums import JobStage
from dlrover.python.rl.common.rl_context import RLContext


class JobContext(Singleton, PickleSerializable):
    """
    JobContext includes all the key runtime information.
    """

    def __init__(self):
        self._job_config = None
        self._rl_context = None
        self._execution_graph = None

        self._job_stage = JobStage.INIT
        # will retry(invoke 'fit') trainer only(without restart other
        # workloads) if this param is set to True
        self._is_trainer_recoverable = False

        self._locker = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_locker" in state:
            del state["_locker"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._locker = threading.Lock()

    def init(self, job_config: JobConfig, rl_context: RLContext):
        self._job_config = job_config
        self._rl_context = rl_context

    @property
    def job_config(self) -> JobConfig:
        return self._job_config

    @property
    def rl_context(self) -> RLContext:
        return self._rl_context

    @property
    def execution_graph(self) -> RLExecutionGraph:
        return self._execution_graph

    def set_execution_graph(self, execution_graph: RLExecutionGraph):
        self._execution_graph = execution_graph

    def set_trainer_recoverable(self):
        self._is_trainer_recoverable = True

    def is_trainer_recoverable(self):
        return self._is_trainer_recoverable


def get_job_context() -> JobContext:
    job_context = JobContext.singleton_instance()
    return job_context
