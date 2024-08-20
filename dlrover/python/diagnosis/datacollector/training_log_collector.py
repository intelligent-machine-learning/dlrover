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

from dlrover.python.diagnosis.common.diagnosis_data import TrainingLog
from dlrover.python.diagnosis.datacollector.data_collector import DataCollector
from dlrover.python.util.file_util import read_last_n_lines


class TrainingLogCollector(DataCollector):
    """
    TrainingLogCollector collects the last n_line lines
    from the given logs.
    """

    def __init__(self, log_file: str = "", n_line: int = 0):
        super().__init__()
        self._log_file = log_file
        self._n_line = n_line

    def collect_data(self) -> TrainingLog:
        if len(self._log_file) == 0:
            return TrainingLog()
        byte_logs = read_last_n_lines(self._log_file, self._n_line)
        logs = [str(line) for line in byte_logs]
        training_log = TrainingLog(logs=logs)
        return training_log

    def to_collect_data(self) -> bool:
        return True
