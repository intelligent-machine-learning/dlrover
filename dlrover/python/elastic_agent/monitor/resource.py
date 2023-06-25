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

import os
import threading
import time

import psutil

from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import singleton
from dlrover.python.elastic_agent.master_client import GlobalMasterClient


def get_process_cpu_percent():
    """Get the cpu percent of the current process."""
    try:
        procTotalPercent = 0
        result = {}
        proc_info = []
        # 分析依赖文件需要获取 memory_maps
        # 使用进程占用的总CPU计算系统CPU占用率
        for proc in psutil.process_iter(
            ["pid", "ppid", "name", "username", "cmdline"]
        ):
            proc_percent = proc.cpu_percent()
            procTotalPercent += proc_percent
            proc.info["cpu_percent"] = round(proc_percent, 2)
            proc_info.append(proc.info)
        # 暂时不上报进程数据，看下数据量的情况
        result["proc_info"] = proc_info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = round(procTotalPercent / cpu_count, 2)
    except Exception:
        cpu_percent = 0.0
    return cpu_percent / 100.0


def get_used_memory():
    """ "Get the used memory of the container"""
    mem = psutil.virtual_memory()
    return int(mem.used / 1024 / 1024)


@singleton
class ResourceMonitor(object):
    def __init__(self):
        """
        The monitor samples the used memory and cpu percent
        reports the used memory and cpu percent to the DLRover master.
        """
        self._total_cpu = psutil.cpu_count(logical=True)
        if (
            os.getenv(NodeEnv.DLROVER_MASTER_ADDR, "")
            and os.getenv(NodeEnv.AUTO_MONITOR_WORKLOAD, "") == "true"
        ):
            threading.Thread(
                target=self._monitor_resource,
                name="monitor_resource",
                daemon=True,
            ).start()

    def start_monitor_cpu(self):
        get_process_cpu_percent()

    def report_resource(self):
        try:
            used_mem = get_used_memory()
            cpu_percent = get_process_cpu_percent()
            current_cpu = round(cpu_percent * self._total_cpu, 2)
            GlobalMasterClient.MASTER_CLIENT.report_used_resource(
                used_mem, current_cpu
            )
            logger.debug(
                "Report Resource CPU : %s, Memory %s", current_cpu, used_mem
            )
        except Exception as e:
            logger.info(e)

    def _monitor_resource(self):
        logger.info("Start to monitor resource usage")
        while True:
            self.report_resource()
            time.sleep(15)
