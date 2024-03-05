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
import pynvml

from dlrover.python.common.constants import NodeEnv
from dlrover.python.common.grpc import GPUStats
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import Singleton
from dlrover.python.elastic_agent.master_client import MasterClient


def get_process_cpu_percent():
    """Get the cpu percent of the current process."""
    try:
        procTotalPercent = 0
        result = {}
        proc_info = []
        for proc in psutil.process_iter(
            ["pid", "ppid", "name", "username", "cmdline"]
        ):
            proc_percent = proc.cpu_percent()
            procTotalPercent += proc_percent
            proc.info["cpu_percent"] = round(proc_percent, 2)
            proc_info.append(proc.info)
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


def get_gpu_stats(gpus=[]):
    """ "Get the used gpu info of the container"""
    if not gpus:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            logger.warning("No GPU is available.")
            device_count = 0
        gpus = list(range(device_count))
    gpu_stats: list[GPUStats] = []
    for i in gpus:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / (1024**2)
        used_memory = memory_info.used / (1024**2)

        # Get GPU utilization
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = utilization.gpu

        gpu_stats.append(
            GPUStats(
                index=i,
                total_memory_mb=total_memory,
                used_memory_mb=used_memory,
                gpu_utilization=gpu_utilization,
            )
        )
    return gpu_stats


class ResourceMonitor(Singleton):
    def __init__(self):
        """
        The monitor samples the used memory and cpu percent
        reports the used memory and cpu percent to the DLRover master.
        """
        self._total_cpu = psutil.cpu_count(logical=True)
        self._gpu_enabled = False
        self._gpu_stats: list[GPUStats] = []
        self._master_client = MasterClient.singleton_instance()

    def start(self):
        if not os.getenv(NodeEnv.DLROVER_MASTER_ADDR, ""):
            return

        self.init_gpu_monitor()
        logger.info("Resource Monitor Initializing ...")

        try:
            thread = threading.Thread(
                target=self._monitor_resource,
                name="monitor_resource",
                daemon=True,
            )
            thread.start()
            if thread.is_alive():
                logger.info("Resource Monitor initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to start the monitor resource thread. Error: {e}"
            )

        # The first time called cpu_percent will return a meaningless 0.0
        # value which we are supposed to ignore. So, here we call it at
        # the begining of monitor and the next value is valid.
        get_process_cpu_percent()

    def stop(self):
        if self._gpu_enabled:
            self.shutdown_gpu_monitor()

    def __del__(self):
        if self._gpu_enabled:
            self.shutdown_gpu_monitor()

    def init_gpu_monitor(self):
        try:
            pynvml.nvmlInit()
            self._gpu_enabled = True
        except pynvml.NVMLError_LibraryNotFound:
            logger.warn(
                "NVIDIA NVML library not found. "
                "GPU monitoring features will be disabled."
            )
        except pynvml.NVMLError_Unknown as e:
            logger.error(
                f"An unknown error occurred during NVML initializing: {e}"
            )
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during NVML initializing: {e}"
            )

    def shutdown_gpu_monitor(self):
        try:
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during NVML shutdown: {e}"
            )

    def report_resource(self):
        try:
            used_mem = get_used_memory()
            cpu_percent = get_process_cpu_percent()
            if self._gpu_enabled:
                self._gpu_stats = get_gpu_stats()
            current_cpu = round(cpu_percent * self._total_cpu, 2)
            self._master_client.report_used_resource(
                used_mem, current_cpu, self._gpu_stats
            )
            logger.debug(
                "Report Resource CPU : %s, Memory %s, GPU %s",
                current_cpu,
                used_mem,
                self._gpu_stats,
            )
        except Exception as e:
            logger.exception(e)

    def _monitor_resource(self):
        logger.info("Start to monitor resource usage")
        while True:
            self.report_resource()
            time.sleep(15)
