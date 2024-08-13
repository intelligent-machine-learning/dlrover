import copy
from typing import Dict
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure
from torch.distributed.elastic.agent.server.api import (
    WorkerSpec,
    RunResult,
)


class WorkerContext:
    def __init__(self, worker_spec: WorkerSpec, remaining_failovers: int, restart_count: int, run_result: RunResult):
        self.worker_spec: WorkerSpec = copy.deepcopy(worker_spec)
        self.remaining_failovers = remaining_failovers
        self.restart_count = restart_count
        self.run_result = copy.deepcopy(run_result)
