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

from dlrover.python.diagnosis.common.diagnosis_data import ChipMetrics
from dlrover.python.diagnosis.datacollector.data_collector import DataCollector


class MetricsCollector(DataCollector):
    def __init__(self, *args, **kwargs):
        """
        MetricsCollector collects GPU metrics
        """
        pass

    def collect_data(self) -> object:
        chip_metrics = ChipMetrics(0)
        return chip_metrics

    def to_collect_data(self) -> bool:
        return True
