# Copyright 2022 The DLRover Authors. All rights reserved.
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

import json
import os
import threading
import time

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import singleton
from dlrover.python.elastic_agent.master_client import MasterClient
from dlrover.python.elastic_agent.monitor.resource import ResourceMonitor


def is_tf_chief():
    TF_CONFIG = os.getenv("TF_CONFIG", "")
    if not TF_CONFIG:
        return False
    config = json.loads(TF_CONFIG)
    task = config.get("task", None)
    if not task:
        return False
    if task.get("type", None) == "chief" and task.get("index", None) == 0:
        return True
    return False


@singleton
class TFTrainingProcessReporter(object):
    def __init__(self):
        self._resource_monitor = ResourceMonitor()
        self._last_timestamp = 0
        self._start_time = 0
        self.called_in_tf_hook = False
        self._is_tf_chief = is_tf_chief()
        self._master_client = MasterClient.singleton_instance()

    def set_start_time(self):
        if self._start_time == 0:
            self._resource_monitor.start()
            timestamp = int(time.time())
            self._last_timestamp = timestamp
            self._start_time = timestamp
            self._resource_monitor.start_monitor_cpu()
            logger.info(
                "Start training process reporter in training hooks : %s",
                self.called_in_tf_hook,
            )

    def report_resource_with_step(self, step):
        if not self._is_tf_chief:
            return
        try:
            timestamp = int(time.time())
            if step > 0 and timestamp - self._last_timestamp > 30:
                self._resource_monitor.report_resource()
                logger.info("Report global step = {}".format(step))
                self._last_timestamp = timestamp
                self._master_client.report_global_step(
                    step,
                    self._last_timestamp,
                )
        except Exception as e:
            logger.warning(e)


@singleton
class TorchTrainingMonitor(object):
    def __init__(self, metrics_path):
        self._resource_monitor = ResourceMonitor()
        self._last_timestamp = 0
        self._start_time = 0
        self._master_client = MasterClient.singleton_instance()
        self._group_rank = env_utils.get_node_rank()
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        self._metrics_path = metrics_path

    def start(self):
        self._resource_monitor.start()
        self._resource_monitor.start_monitor_cpu()
        thread = threading.Thread(
            target=self._periodically_report_step,
            name="report-step",
            daemon=True,
        )
        thread.start()

    def stop(self):
        self._resource_monitor.stop()

    def report_resource_with_step(self):
        if self._group_rank != 0:
            return
        try:
            if not os.path.exists(self._metrics_path):
                return
            with open(self._metrics_path, "r") as f:
                record = json.load(f)
                step = record.get("step", 0)
                timestamp = record.get("timestamp", 0)
            if step > 0 and timestamp - self._last_timestamp > 15:
                self._resource_monitor.report_resource()
                logger.info("Report global step = {}".format(step))
                self._last_timestamp = timestamp
                self._master_client.report_global_step(
                    step,
                    self._last_timestamp,
                )
        except Exception as e:
            logger.warning(e)

    def _periodically_report_step(self):
        if self._group_rank != 0:
            return
        while True:
            self.report_resource_with_step()
            time.sleep(15)
