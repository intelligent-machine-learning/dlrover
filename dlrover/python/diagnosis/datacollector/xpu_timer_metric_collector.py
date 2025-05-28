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

import requests

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisDataType,
    EnvConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.diagnosis.datacollector.data_collector import DataCollector
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.util.common_util import is_port_in_use


class XpuTimerMetricsCollector(DataCollector):
    def __init__(self):
        """
        MetricsCollector collects GPU metrics from xpu-timer.
        """
        super().__init__()
        self._metric_port = env_utils.get_env(EnvConfigKey.XPU_TIMER_PORT)
        if self._metric_port:
            self._metric_endpoint = (
                "http://127.0.0.1:" + self._metric_port + "/metrics"
            )
        else:
            self._metric_endpoint = None
        self._client = MasterClient.singleton_instance()

    def collect_data(self) -> str:
        if not self.is_enabled():
            return ""

        try:
            response = requests.get(self._metric_endpoint)
            response.raise_for_status()

            # data preprocessing
            return self._preprocess_metrics(response.text)
        except requests.exceptions.RequestException as e:
            logger.warning(
                "Error fetching metrics from "
                f"xpu-timer: {self._metric_endpoint}, error: {e}"
            )
            return ""

    def _preprocess_metrics(self, metric_str):
        try:
            metric_list = [
                line
                for line in metric_str.splitlines()
                if not line.startswith("#") and not line.startswith("exposer")
            ]
            return "\n".join(metric_list)
        except Exception as e:
            logger.warning(f"Error preprocessing metrics from xpu-timer: {e}")
            return ""

    def is_enabled(self) -> bool:
        return self._metric_endpoint is not None and is_port_in_use(
            self._metric_port
        )

    def store_data(self, data: object):
        if not isinstance(data, str):
            logger.warning("The data is not of type string")
            return

        agent_xpu_metric = WorkerTrainingMetric(
            data_type=DiagnosisDataType.XPU_TIMER_METRIC,
            data_content=data,
            node_id=env_utils.get_node_id(),
            node_type=env_utils.get_node_type(),
            node_rank=env_utils.get_node_rank(),
        )
        self._client.report_diagnosis_agent_metrics(agent_xpu_metric)
