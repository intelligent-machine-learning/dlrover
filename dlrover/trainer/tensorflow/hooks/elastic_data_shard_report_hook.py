# Copyright 2023 The DLRover Authors. All rights reserved.
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

from tensorflow.python.training.session_run_hook import SessionRunHook

from dlrover.trainer.util.log_util import default_logger as logger


class ElasticDataShardReportHook(SessionRunHook):
    def __init__(self, sharding_client):
        self._sharding_client = sharding_client

    def after_run(self, run_context, run_values):
        try:
            self._sharding_client.report_batch_done()
            logger.info("report_batch_done")
        except Exception as ex:
            logger.error("DLrover agent: report batch done failed: %s", ex)
