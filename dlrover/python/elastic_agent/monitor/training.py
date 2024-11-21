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
import re
import threading
import time
from datetime import datetime

from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
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


class TFTrainingReporter(Singleton):
    def __init__(self):
        self._resource_monitor = ResourceMonitor.singleton_instance()
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


class TorchTrainingMonitor(Singleton):
    def __init__(
        self, metrics_path, logfile="", match_pattern="", step_rank=0
    ):
        self._resource_monitor = ResourceMonitor.singleton_instance()
        self._last_timestamp = 0
        self._start_time = 0
        self._master_client = MasterClient.singleton_instance()
        self._group_rank = env_utils.get_node_rank()
        self._rank = env_utils.get_rank()
        self._world_size = env_utils.get_world_size()
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        self._metrics_path = metrics_path
        self._user_step_logfile = logfile
        self._user_step_pattern = match_pattern
        self._user_step_rank = (
            self._world_size + step_rank
        ) % self._world_size
        self.stop_step_collector = False

    def start(self):
        if os.getenv(NodeEnv.MONITOR_ENABLED, "false") != "true":
            logger.info(
                f"Skip starting monitor for {NodeEnv.MONITOR_ENABLED} "
                "disabled."
            )
            return
        self._resource_monitor.start()
        thread = threading.Thread(
            target=self._periodically_report,
            name="node_reporter",
            daemon=True,
        )
        thread.start()

        _user_step_reporter = threading.Thread(
            target=self._user_step_report,
            name="user_step_reporter",
            daemon=True,
        )
        _user_step_reporter.start()

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
                self._last_timestamp = timestamp
                self._master_client.report_global_step(
                    step,
                    self._last_timestamp,
                )
        except Exception as e:
            logger.warning(e)

    def send_heartbeat(self):
        try:
            ts = int(time.time())
            action = self._master_client.report_heart_beat(ts)
            if action:
                pass
        except Exception:
            logger.warning("Fail to report a heartbeat.")

    def _periodically_report(self):
        logger.info("Start training agent reporter.")
        while True:
            if self._group_rank == 0:
                self.report_resource_with_step()
            self.send_heartbeat()
            time.sleep(15)

    def do_user_step_collect(self, _step_logfile, _step_pattern):
        try:
            with open(_step_logfile, "r") as f:
                logger.info(f"try to collect steps from {_step_logfile}")
                f.seek(0, 0)
                expr = re.compile(_step_pattern)
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(2)
                        continue

                    m = expr.match(line)
                    if m is not None:
                        dt_obj = datetime.strptime(
                            m.groups()[0], "%Y-%m-%d %H:%M:%S"
                        )
                        ts = int(dt_obj.timestamp())
                        step = int(m.groups()[1])
                        total_step = int(m.groups()[2])
                        logger.info(
                            f"Report step to master: {step}/{total_step} {ts}"
                        )
                        self._master_client.report_user_step(
                            ts, step, total_step
                        )

                    if self.stop_step_collector:
                        break
        except Exception as e:
            logger.warning(f"failed to collect user steps: {str(e)}")

    def _user_step_report(self):
        logger.info(
            f"step report: {self._user_step_logfile} {self._user_step_pattern}"
        )
        if self._user_step_logfile == "" or self._user_step_pattern == "":
            logger.info(
                "stop step report because of invalid logfile or pattern"
            )
            return
        myrank = env_utils.get_rank()
        logger.info(
            f"my rank is {myrank}, step rank is {self._user_step_rank}"
        )
        if myrank != self._user_step_rank:
            logger.info(f"Rank {myrank} skip user step reporting")
            return

        logger.info("Start user step reporter.")
        while True:
            self.do_user_step_collect(
                self._user_step_logfile, self._user_step_pattern
            )
            time.sleep(10)
