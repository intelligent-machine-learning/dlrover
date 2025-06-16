from dataclasses import dataclass
from enum import Enum
import time


from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
)
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.new_master.job_manager import JobManager


@dataclass
class RestartInfo:
    restart_time: int = 0
    with_failover: bool = True
    reason: str = ""


class JobStage(str, Enum):
    INIT = "INIT"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class Core:
    def __init__(self, job_config: JobConfig) -> None:
        self.job_config = job_config

        self.stage = JobStage.INIT

        self.rdzv_manager = ElasticTrainingRendezvousManager()
        self.job_manager = JobManager(self)

        self.job_manager.init_nodes()

    def run(self):
        if self.stage == JobStage.INIT:
            self.job_manager.create_nodes()
            self.job_manager.precheck_nodes()
            self.job_manager.start_job()
            self.stage = JobStage.RUNNING
        while self.stage == JobStage.RUNNING:
            self.job_manager.monitor_nodes()
            # Save state or perform other periodic tasks
            time.sleep(5)
        if self.stage != JobStage.STOPPED:
            self.stop()

    def stop(self):
        if self.stage == JobStage.STOPPING or self.stage == JobStage.STOPPED:
            return
        self.stage = JobStage.STOPPING
        self.job_manager.stop_job()
