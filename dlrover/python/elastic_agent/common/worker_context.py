# Copyright 2024 The DLRover Authors. All rights reserved.
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

from typing import List, Optional

from torch.distributed.elastic.agent.server.api import RunResult, WorkerSpec

from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    DiagnosisActionQueue,
)
from dlrover.python.common.singleton import Singleton
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    DiagnosisActionConstants,
)


class WorkerContext(Singleton):
    def __init__(self):
        self._worker_spec: Optional[WorkerSpec] = None
        self.remaining_failovers = 0
        self.restart_count = 0
        self._run_result: Optional[RunResult] = None
        self._diagnose_action_queue = DiagnosisActionQueue()

    @property
    def worker_spec(self):
        return self._worker_spec

    @property
    def run_result(self):
        return self._run_result

    def to_string(self) -> str:
        return (
            "WorkerContext:\n"
            f"worker_spec: {self._worker_spec}\n"
            f"remaining_failover: {self.remaining_failovers}\n"
            f"restart_count: {self.restart_count}\n"
            f"run_result: {self._run_result}"
        )

    def _update_context(
        self,
        worker_spec: WorkerSpec = None,
        remaining_failovers: int = 0,
        restart_count: int = 0,
        run_result: RunResult = None,
    ):
        self._worker_spec: WorkerSpec = worker_spec
        self.remaining_failovers = remaining_failovers
        self.restart_count = restart_count
        self._run_result = run_result

    def enqueue_diagnose_action(self, action: DiagnosisAction):
        self._diagnose_action_queue.add_action(action)

    def next_actions(
            self,
            instance=DiagnosisConstant.LOCAL_INSTANCE,
            action_type=DiagnosisActionConstants.ACTION_TYPE_ANY,
    ) -> List[DiagnosisAction]:
        return self._diagnose_action_queue.next_actions(
            instance=instance, action_type=action_type
        )

def get_worker_context() -> WorkerContext:
    return WorkerContext.singleton_instance()

def update_worker_context(
    worker_spec: WorkerSpec = None,
    remaining_failovers: int = 0,
    restart_count: int = 0,
    run_result: RunResult = None,
):
    worker_context = get_worker_context()
    worker_context._update_context(worker_spec, remaining_failovers, restart_count, run_result)
