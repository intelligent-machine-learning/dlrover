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
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.constant import (
    DLJobExitReason,
    DLMasterConstant,
)
from dlrover.python.unified.common.dl_context import DLContext
from dlrover.python.unified.common.enums import JobStage
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.common.job_context import (
    JobContext,
    get_job_context,
)
from dlrover.python.unified.master.state_backend import (
    MasterStateBackendFactory,
)
from dlrover.python.unified.remote.call_obj import RuntimeInfo


class BaseMaster(ABC):
    """
    DLRover Master is the core of the control plane management for the entire
    DL training. It is responsible for all related control plane management
    operations.
    """

    def __init__(self, job_config_serialized, dl_context_serialized):
        # init job context
        self._job_config = JobConfig.deserialize(job_config_serialized)
        self._dl_context = DLContext.deserialize(dl_context_serialized)
        self._job_context = get_job_context()
        self._job_context.init(self._job_config, self._dl_context)

        # init state backend
        self._state_backend = MasterStateBackendFactory.get_state_backend()
        self._state_backend.init()

        self._create_time = int(time.time())
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.Lock()

        self._job_manager = None

        logger.info(
            f"DLRover Master: {({self.__class__.__name__})} initiated with "
            f"job-config: {self._job_config}, "
            f"dl-context: {self._dl_context}."
        )

    @abstractmethod
    def init(self):
        """To initialize the master core components."""

    @abstractmethod
    def _handle_failures(self, failures: List[FailureDesc]):
        """To handle failures."""

    @property
    def context(self):
        return self._job_context

    @property
    def job_name(self):
        return self._job_context.job_config.job_name

    @property
    def state_backend(self):
        return self._state_backend

    @property
    def job_manager(self):
        return self._job_manager

    def _update_job_stage(self, stage):
        self._job_context.set_job_stage(stage)

    def get_job_stage(self):
        return self.context.get_job_stage()

    def is_job_started(self) -> bool:
        return self.get_job_stage() != JobStage.INIT

    def _get_job_context_state_key(self):
        key = (
            self.context.job_config.job_name
            + "_"
            + DLMasterConstant.JOB_CONTEXT_STATE_KEY
        )
        return key.encode()

    def _load_context_from_checkpoint(self):
        context_key = self._get_job_context_state_key()

        if self.state_backend.exists(context_key):
            return JobContext.deserialize(self.state_backend.load(context_key))
        return None

    def _save_context_to_checkpoint(self):
        context_key = self._get_job_context_state_key()
        self.state_backend.set(context_key, self.context.serialize())
        logger.info("Job context saved checkpoint.")

    def _cleanup_context_checkpoint(self):
        context_key = self._get_job_context_state_key()
        if self.state_backend.exists(context_key):
            self.state_backend.delete(context_key)
            logger.info("Job context cleanup checkpoint.")

    def update_job_context(self):
        # TODO: impl
        self._save_context_to_checkpoint()

    def _gen_master_failure(self):
        if self.context.is_trainer_recoverable():
            failure_level = 1
        else:
            failure_level = 2

        return FailureDesc(
            failure_obj="MASTER",
            failure_time=self._create_time,
            failure_level=failure_level,
            reason="DMaster unexpected exit.",
        )

    def run(self):
        self._update_job_stage(JobStage.RUNNING)

        try:
            # load context from backend
            context_from_ckpt = self._load_context_from_checkpoint()
            if context_from_ckpt:
                logger.info(
                    "DLMaster recover from checkpoint, "
                    f"context: {context_from_ckpt}."
                )
                self._job_context = context_from_ckpt
                self._handle_failures(self._gen_master_failure())
            else:
                logger.info(f"DLMaster new job executing: {self.job_name}.")
                self._save_context_to_checkpoint()
                self.job_manager.start_job()

            self._executor.submit(self._wait_and_exit)
        except Exception as e:
            logger.error("Got unexpected fatal error on starting.", e)
            self.exit_job(
                stage=JobStage.ERROR, forced=True, reason=DLJobExitReason.ERROR
            )

    def _get_master_wait_interval(self):
        # default
        return DLMasterConstant.RUN_WAIT_INTERVAL

    def _wait_and_exit(self):
        while True:
            if self.get_job_stage() == JobStage.ERROR:
                logger.info("Exit loop for job in error stage.")
                break

            if self._job_manager.is_job_finished():
                if self.get_job_stage() == JobStage.RUNNING:
                    self.exit_job(reason=DLJobExitReason.FINISHED)
                else:
                    logger.info(
                        "Job manager is finished but job stage not "
                        f"running: {self.get_job_stage()}, "
                        "keep waiting..."
                    )
            # TODO: temp impl for now
            elif self._job_manager.has_job_error():
                logger.warning("Job got error, try failover...")
                self._handle_failures(
                    self._job_manager.gen_failures_by_error()
                )

            time.sleep(self._get_master_wait_interval())
            logger.debug("DLMaster still running...")

    def _should_continue_exiting(self) -> Tuple[bool, str]:
        # check failure handling
        if self.context.is_in_failover():
            return False, "job in failover"

        return True, ""

    def exit_job(self, stage=None, need_cleanup=True, forced=False, reason=""):
        logger.info(
            f"DLMaster exit job with cleanup: {need_cleanup}, "
            f"reason: {reason}."
        )
        if not stage:
            self._update_job_stage(JobStage.FINISHED)
        else:
            self._update_job_stage(stage)

        interval = DLMasterConstant.EXIT_WAIT_INTERVAL
        for i in range(interval):
            logger.info(f"DLMaster will cleanup and exit in {interval - i}s.")
            should_continue, reason = self._should_continue_exiting()
            if not forced and not should_continue:
                logger.info(f"DLMaster exiting is interrupted by: {reason}.")
                return
            time.sleep(1)

        if need_cleanup:
            logger.info("DLMaster do cleanup for exiting.")
            if self._job_manager:
                self._job_manager.stop_job()

        self._cleanup_context_checkpoint()

        logger.info("DLMaster exit now.")
        ray.kill(self)

    """Remote call functions start"""

    def ping(self):
        logger.debug("Ping called.")
        return True

    def get_job_status(self):
        return self.get_job_stage().name

    def report_restarting(
        self,
        name: str,
        timestamp: int,
        level: int = -1,
        reason: str = "",
        **kwargs,
    ):
        vertex_role = self.context.execution_graph.name_vertex_mapping[
            name
        ].role
        failure_desc = FailureDesc(
            failure_obj="WORKLOAD",
            workload_name=name,
            workload_role=vertex_role,
            failure_time=timestamp,
            failure_level=level,
            reason=reason,
            extra_info=kwargs,
        )
        self._handle_failures([failure_desc])

    def report_runtime(self, runtime_info: RuntimeInfo):
        if not runtime_info:
            return

        name = runtime_info.name

        name_vertex_mapping = (
            self._job_context.execution_graph.name_vertex_mapping
        )
        if name in name_vertex_mapping:
            # update runtime info
            name_vertex_mapping[name].update_runtime_info(
                create_time=runtime_info.create_time,
                hostname=runtime_info.hostname,
                host_ip=runtime_info.host_ip,
            )
        logger.debug(f"Got runtime info: {runtime_info} reported by: {name}.")

    def report_failure(self, failure: FailureDesc):
        self._handle_failures([failure])

    """Remote call functions end"""
