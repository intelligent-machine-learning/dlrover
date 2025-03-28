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

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import DiagnosisErrorConstant
from dlrover.python.diagnosis.common.diagnostician import (
    DiagnosisObservation,
    Diagnostician,
)
from dlrover.python.diagnosis.datacollector.training_log_collector import (
    TrainingLogCollector,
)


class FailureNodeDiagnostician(Diagnostician):
    """
    FailureNodeDiagnostician is to observe and resolve the failure node problem
    """

    def __init__(self):
        super().__init__()

    def observe(self, **kwargs) -> DiagnosisObservation:
        log_file_arg = kwargs.get("log_file")
        if log_file_arg is None or not isinstance(log_file_arg, str):
            logger.error(f"Invalid log_file: {log_file_arg}")
            return DiagnosisObservation()
        log_file = str(log_file_arg)

        errors_arg = kwargs.get("errors")
        if errors_arg is None or not isinstance(errors_arg, str):
            logger.error(f"Invalid errors: {errors_arg}")
            return DiagnosisObservation()
        errors = str(errors_arg)
        # temp usage: express the env for specified error info
        # e.g.
        # export FAILURE_NODE_ERRORS="#error code is 12345# error code is
        # 23456# error code is 507035#"
        error_codes = errors.split("#")
        error_codes = [error_code.strip() for error_code in error_codes]

        collector = TrainingLogCollector(log_file, 5000)
        training_log = collector.collect_data()
        logs = training_log.logs
        if not logs or len(logs) == 0:
            logger.warning(f"fail to collect training logs from {log_file}")
            return DiagnosisObservation()

        is_failure_node = False
        for log in logs:
            if is_failure_node:
                break
            for error in error_codes:
                if len(error) > 0 and "#" not in log and error in log:
                    logger.info(
                        f"Got #{error}# in {log}, set as failure node."
                    )
                    is_failure_node = True
                    break
        if is_failure_node:
            return DiagnosisObservation(
                observation=DiagnosisErrorConstant.NODE_FAILED,
            )
        return DiagnosisObservation()
