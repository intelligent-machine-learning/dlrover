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
from concurrent.futures import ThreadPoolExecutor

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.constant import RLMasterConstant
from dlrover.python.rl.common.job_context import JobContext, get_job_context
from dlrover.python.rl.common.rl_context import RLContext
from dlrover.python.rl.master.job_manager import JobManager
from dlrover.python.rl.master.state_backend import MasterStateBackendFactory
from dlrover.python.rl.remote.call_obj import RuntimeInfo


@ray.remote
class DLRoverRLMaster(object):
    def __init__(self, job_config_serialized, rl_context_serialized):
        # init job context
        self._job_config = JobConfig.deserialize(job_config_serialized)
        self._rl_context = RLContext.deserialize(rl_context_serialized)
        self._job_context = get_job_context()
        self._job_context.init(self._job_config, self._rl_context)

        # init state backend
        self._state_backend = MasterStateBackendFactory.get_state_backend()
        self._state_backend.init()

        self._job_manager = JobManager()

        self._started = False
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            "DLRover RLMaster initiated with "
            f"job-config: {self._job_config}, "
            f"rl-context: {self._rl_context}."
        )

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

    def is_started(self) -> bool:
        return self._started

    def _get_job_context_state_key(self):
        key = (
            self.context.job_config.job_name
            + "_"
            + RLMasterConstant.JOB_CONTEXT_STATE_KEY
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

    def run(self):
        self._started = True

        try:
            # load context from backend
            context_from_ckpt = self._load_context_from_checkpoint()
            if context_from_ckpt:
                logger.info(
                    "RLMaster recover from checkpoint, "
                    f"context: {context_from_ckpt}."
                )
                self._job_context = context_from_ckpt
                # TODO: recover logic
            else:
                logger.info(f"RLMaster new job executing: {self.job_name}.")
                self._save_context_to_checkpoint()
                self.job_manager.start_job()

            self._executor.submit(self._wait_and_exit)
        except Exception as e:
            logger.info("Got unexpected exception while running.", e)
            self.exit_job()

    def _wait_and_exit(self):
        while True:
            if self._job_manager.is_job_finished():
                self.exit_job()
            time.sleep(RLMasterConstant.RUN_WAIT_INTERVAL)
            logger.info("RLMaster still running...")

    def exit_job(self, need_cleanup=True):
        logger.info(f"RLMaster exit job with cleanup: {need_cleanup}.")
        self._started = False

        if need_cleanup:
            if self._job_manager:
                self._job_manager.stop_job()

        self._cleanup_context_checkpoint()

        interval = RLMasterConstant.EXIT_WAIT_INTERVAL
        for i in range(interval):
            logger.info(f"RLMaster will exit in {interval - i}s.")
            time.sleep(1)
        logger.info("RLMaster exit now.")
        ray.actor.exit_actor()

    """Remote call functions start"""

    def ping(self):
        return True

    def report(self, runtime_info: RuntimeInfo):
        if not runtime_info:
            return

        is_actor_failover = False
        name = runtime_info.name

        logger.info(f"Got runtime info: {runtime_info} reported by: {name}.")
        name_vertex_mapping = (
            self._job_context.execution_graph.name_vertex_mapping
        )
        if name in name_vertex_mapping:
            vertex = name_vertex_mapping[name]

            if vertex.create_time:
                is_actor_failover = True

            # update runtime info
            name_vertex_mapping[name].update_runtime_info(
                create_time=runtime_info.create_time,
                hostname=runtime_info.hostname,
                host_ip=runtime_info.host_ip,
            )

        # deal with failover
        if is_actor_failover:
            # TODO
            pass

    """Remote call functions end"""
