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

import os
import threading
import time

import psutil

from dlrover.python.common.comm import GPUStats
from dlrover.python.common.constants import NodeEnv, Accelerators
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
    """Get the used memory of the container"""
    mem = psutil.virtual_memory()
    return int(mem.used / 1024 / 1024)


def get_gpu_stats(gpus=[]):
    """Get the used gpu info of the container"""

    try:
        import pynvml
    except ImportError:
        logger.warning("No pynvml is available, skip getting gpu stats.")
        return []

    try:
        pynvml.nvmlInit()

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
    except pynvml.NVMLError_LibraryNotFound:
        logger.debug("Not nv environment, skip getting gpu stats.")
        return []
    except Exception as e:
        logger.warning(f"Got unexpected error when getting gpu stats: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_hpu_stats(hpus=[]):
    """Get the used hpu info of the container"""

    try:
        import acl
    except ImportError:
        logger.warning("No Ascend acl is available, skip getting hpu stats.")
        return []

    try:
        acl.init()
        if not hpus:
            try:
                device_count = acl.rt.get_device_count()[0]
            except Exception:
                logger.warning("No HPU is available.")
                device_count = 0
            hpus = list(range(device_count))
        hpu_stats: list[GPUStats] = []
        for i in hpus:
            acl.rt.set_device(i)
            # get HPU memory
            # format: (free, total, ret)
            free_mem, total_mem, ret = acl.rt.get_mem_info(0)
            if ret != 0:
                raise RuntimeError("Failed to execute get_mem_info")
            used_mem = total_mem - free_mem

            # get HPU utilization
            # format: ({'cube_utilization': 0, 'vector_utilization': 0, 'aicpu_utilization': 0, 'memory_utilization': -1, 'utilization_extend': 0}, ret)
            utilization_dict, ret = acl.rt.get_device_utilization_rate(i)
            if ret != 0:
                raise RuntimeError(
                    "Failed to execute get_device_utilization_rate"
                )
            # use cube_utilization as common gpu_utilization
            utilization = utilization_dict["cube_utilization"]

            hpu_stats.append(
                GPUStats(
                    index=i,
                    total_memory_mb=total_mem,
                    used_memory_mb=used_mem,
                    gpu_utilization=utilization,
                )
            )
        return hpu_stats
    except Exception as e:
        logger.warning(f"Got unexpected error when getting hpu stats: {e}")
        return []
    finally:
        acl.finalize()


class ResourceMonitor(Singleton):
    def __init__(self, gpu_type: str = Accelerators.NVIDIA_GPU):
        """
        The monitor samples the used memory and cpu percent
        reports the used memory and cpu percent to the DLRover master.
        """
        self._total_cpu = psutil.cpu_count(logical=True)
        self._gpu_type = gpu_type
        self._gpu_stats: list[GPUStats] = []
        self._master_client = MasterClient.singleton_instance()

        if os.getenv(NodeEnv.DLROVER_MASTER_ADDR, ""):
            # The first time called cpu_percent will return a meaningless 0.0
            # value which we are supposed to ignore. So, here we call it at
            # the beginning of monitor and the next value is valid.
            get_process_cpu_percent()

    def start(self):
        if not os.getenv(NodeEnv.DLROVER_MASTER_ADDR, ""):
            return

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

    def stop(self):
        pass

    def report_resource(self):
        used_mem = get_used_memory()
        cpu_percent = get_process_cpu_percent()

        if self._gpu_type == Accelerators.NVIDIA_GPU:
            self._gpu_stats = get_gpu_stats()
        elif self._gpu_type == Accelerators.ASCEND_NPU:
            self._gpu_stats = get_hpu_stats()
        else:
            # not supported for others
            pass

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

    def _monitor_resource(self):
        logger.info("Start to monitor resource usage")
        while True:
            try:
                self.report_resource()
            except Exception as e:
                logger.debug(f"report resource error: {e}")
            time.sleep(15)
