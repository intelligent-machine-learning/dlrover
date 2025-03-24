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
import os

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.config import JobConfig
from dlrover.python.rl.common.constant import RLMasterConstant
from dlrover.python.rl.common.context import (
    JobContext,
    RLContext,
    get_job_context,
)
from dlrover.python.rl.master.job_manager import JobManager
from dlrover.python.rl.master.state_backend import MasterStateBackendFactory


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

        logger.infof(
            "DLRover RLMaster initiated with "
            f"job-config: {self._job_config}, "
            f"rl-context: {self._rl_context}."
        )
        self.start()

    @property
    def context(self):
        return self._job_context

    @property
    def state_backend(self):
        return self._state_backend

    @property
    def job_manager(self):
        return self._job_manager

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

    def update_job_context(self):
        # TODO: impl
        self._save_context_to_checkpoint()

    def start(self):
        self._started = True

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
            logger.info("RLMaster new job executing.")
            self.job_manager.start()

    def stop(self):
        self._started = False

        # TODO: cleanup

        os._exit(0)
