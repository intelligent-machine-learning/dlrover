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

from torch.distributed.elastic.agent.server.api import RunResult, WorkerSpec
from dlrover.python.diagnosis.common.diagnose_action import (
    DiagnoseActionQueue,
    DiagnoseAction,
)
from typing import Optional, List


class WorkerContext:
    def __init__(self):
        self._worker_spec: Optional[WorkerSpec] = None
        self.remaining_failovers = 0
        self.restart_count = 0
        self._run_result: Optional[RunResult] = None
        self._diagnose_action_queue = DiagnoseActionQueue()

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

    def update_context(
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

    def enqueue_diagnose_action(self, action: DiagnoseAction):
        self._diagnose_action_queue.add_action(action)

    def next_actions(self) -> List[DiagnoseAction]:
        return self._diagnose_action_queue.next_actions()
