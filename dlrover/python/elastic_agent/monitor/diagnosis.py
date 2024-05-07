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

import threading
import time
from typing import Dict

from dlrover.python.common.diagnosis import ChipMetrics, CudaLog, TrainingLog
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.elastic_agent.datacollector.cuda_log_collector import (
    CudaLogCollector,
)
from dlrover.python.elastic_agent.datacollector.data_collector import (
    CollectorType,
    DataCollector,
)
from dlrover.python.elastic_agent.datacollector.log_collector import (
    LogCollector,
)
from dlrover.python.elastic_agent.datacollector.metrics_collector import (
    MetricsCollector,
)
from dlrover.python.elastic_agent.master_client import MasterClient


class DiagnosisMonitor(Singleton):
    def __init__(self):
        self.collectors: dict[str, DataCollector] = {}
        self._master_client = MasterClient.singleton_instance()

    def init(self):
        self.collectors = register_collectors()

    def start(self):
        self.init()
        logger.info("Resource Monitor Initializing ...")

        try:
            thread = threading.Thread(
                target=self._diagnose_faults(),
                name="diagnosis_data",
                daemon=True,
            )
            thread.start()
            if thread.is_alive():
                logger.info("Diagnosis Monitor initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to start the diagnosis monitor thread. Error: {e}"
            )

    def stop(self):
        pass

    def __del__(self):
        pass

    def _diagnose_faults(self):
        logger.info("Start to diagnose faults")
        while True:
            self._collect_diagnosis_data()
            time.sleep(60)

    def _collect_diagnosis_data(self):
        for name, collector in self.collectors:
            if collector.to_collect_data():
                try:
                    data = collector.collect_data()
                    self.report_diagnosis_data(data)
                except Exception as e:
                    logger.error(
                        f"collector {name} fail to collects data: {e}"
                    )

    def report_diagnosis_data(self, data):
        if isinstance(data, CudaLog):
            self._master_client.report_diagnosis_cuda_log(data)
        elif isinstance(data, TrainingLog):
            self._master_client.report_diagnosis_training_log(data)
        elif isinstance(data, ChipMetrics):
            self._master_client.report_diagnosis_chip_metrics(data)
        else:
            raise TypeError(f"invalid data type: {type(data)}")

    def collect_data(self, collector_type: str) -> object:
        if collector_type not in self.collectors:
            return None

        collector = self.collectors[collector_type]
        return collector.collect_data()

    def get_collectors(self):
        return self.collectors


def register_collectors() -> Dict[str, DataCollector]:
    return {
        CollectorType.CUDALOG: CudaLogCollector(),
        CollectorType.CHIPMETRICS: MetricsCollector(),
        CollectorType.TRAININGLOG: LogCollector(),
    }
