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

from abc import ABCMeta, abstractmethod
from typing import Dict

from dlrover.python.common.constants import GpuMetricEnum, NpuMetricEnum


class XpuMetric(metaclass=ABCMeta):
    """XPU Metric
    Attributes:
        type(string): xpu type
    """

    def __init__(self, xpu_type):
        self.type = xpu_type

    @abstractmethod
    def set_metric(self, key, value):
        pass

    @abstractmethod
    def get_metric(self, key):
        pass


class GpuMetric(XpuMetric):
    """GPU Metric
    Attributes:
        gpu_free_mem(int, MB): free gpu memory
        gpu_used_mem(int, MB): used gpu memory
        gpu_util(int): gpu utilization
        gpu_temperature(int): gpu temperature
        gpu_sm_util(float, percent): gpu sm utilization
        gpu_tensor_util(float, percent): gpu tensor utilization
    """

    def __init__(
        self,
        gpu_free_mem=0,
        gpu_used_mem=0,
        gpu_util=0,
        gpu_temperature=0,
        gpu_sm_util=0.0,
        gpu_tensor_util=0.0,
    ):
        super().__init__("nvidia.GPU")
        self.metrics = {
            GpuMetricEnum.GPU_FREE_MEM: gpu_free_mem,
            GpuMetricEnum.GPU_USED_MEM: gpu_used_mem,
            GpuMetricEnum.GPU_UTIL: gpu_util,
            GpuMetricEnum.GPU_TEMP: gpu_temperature,
            GpuMetricEnum.GPU_SM_UTIL: gpu_sm_util,
            GpuMetricEnum.GPU_TENSOR_UTIL: gpu_tensor_util,
        }

    def set_metric(self, key, value):
        if key in self.metrics.keys():
            self.metrics[key] = value

    def get_metric(self, key):
        if key in self.metrics.keys():
            return self.metrics[key]
        else:
            return None


class NpuMetric(XpuMetric):
    """NPU Metric
    Attributes:
        npu_total_mem(int, MB): total npu memory
        npu_used_mem(int, MB): used npu memory
        npu_util(int): npu utilization
        npu_temperature(int): npu temperature
        npu_health_state(0 or 1): npu health state
        npu_link_state(0 or 1): npu link state
        npu_optical_state(0 or 1): npu optical state
        npu_network_state(0 or 1): npu network state
        npu_chip_info_bandwidth_rx(float, MB/s): RDMA rx bandwidth
        npu_chip_info_bandwidth_tx(float, MB/s): RDMA tx bandwidth
    """

    def __init__(
        self,
        npu_total_mem=0,
        npu_used_mem=0,
        npu_util=0,
        npu_temperature=0,
        npu_health_state=1,
        npu_link_state=1,
        npu_optical_state=1,
        npu_network_state=1,
        npu_tx=0.0,
        npu_rx=0.0,
    ):
        super().__init__("ascend.NPU")
        self.metrics = {
            NpuMetricEnum.NPU_TOTAL_MEM: npu_total_mem,
            NpuMetricEnum.NPU_USED_MEM: npu_used_mem,
            NpuMetricEnum.NPU_UTIL: npu_util,
            NpuMetricEnum.NPU_TEMP: npu_temperature,
            NpuMetricEnum.NPU_HEALTH_STATE: npu_health_state,
            NpuMetricEnum.NPU_LINK_STATE: npu_link_state,
            NpuMetricEnum.NPU_OPTICAL_STATE: npu_optical_state,
            NpuMetricEnum.NPU_NETWORK_STATE: npu_network_state,
            NpuMetricEnum.NPU_RDMA_TX: npu_tx,
            NpuMetricEnum.NPU_RDMA_RX: npu_rx,
        }

    def set_metric(self, key, value):
        if key in self.metrics.keys():
            self.metrics[key] = value

    def get_metric(self, key):
        if key in self.metrics.keys():
            return self.metrics[key]
        else:
            return None


class XpuNodeMetric(object):
    """
    Metrics of all XPUs in a single node

    list of XpuMetric with index as local rank id
    """

    def __init__(self):
        pass

    @abstractmethod
    def update_avg_metrics(self):
        pass

    @abstractmethod
    def get_avg_metric(self, metric):
        pass

    @abstractmethod
    def get_node_metrics(self, metric):
        pass


class GpuNodeMetric(XpuNodeMetric):
    """
    Metrics of all GPUs in a single node

    """

    def __init__(self):
        super().__init__()
        self.node_metrics: Dict[int, GpuMetric] = {}
        self.avg_metrics = GpuMetric()

    def get_avg_metric(self, metric):
        return self.avg_metrics.get_metric(metric)

    def get_node_metrics(self, metric):
        metrics = []
        for v in self.node_metrics.values():
            metrics.append(v.get_metric(metric))
        return metrics

    def update_avg_metrics(self):
        self.avg_metrics.metrics[GpuMetricEnum.GPU_FREE_MEM] = 0
        self.avg_metrics.metrics[GpuMetricEnum.GPU_USED_MEM] = 0
        self.avg_metrics.metrics[GpuMetricEnum.GPU_UTIL] = 0
        self.avg_metrics.metrics[GpuMetricEnum.GPU_SM_UTIL] = 0.0
        self.avg_metrics.metrics[GpuMetricEnum.GPU_TENSOR_UTIL] = 0.0

        for _, metric in self.node_metrics.items():
            self.avg_metrics.metrics[
                GpuMetricEnum.GPU_FREE_MEM
            ] += metric.get_metric(GpuMetricEnum.GPU_FREE_MEM)
            self.avg_metrics.metrics[
                GpuMetricEnum.GPU_USED_MEM
            ] += metric.get_metric(GpuMetricEnum.GPU_USED_MEM)
            self.avg_metrics.metrics[
                GpuMetricEnum.GPU_UTIL
            ] += metric.get_metric(GpuMetricEnum.GPU_UTIL)
            self.avg_metrics.metrics[
                GpuMetricEnum.GPU_SM_UTIL
            ] += metric.get_metric(GpuMetricEnum.GPU_SM_UTIL)
            self.avg_metrics.metrics[
                GpuMetricEnum.GPU_TENSOR_UTIL
            ] += metric.get_metric(GpuMetricEnum.GPU_TENSOR_UTIL)

        self.avg_metrics.metrics[GpuMetricEnum.GPU_FREE_MEM] = round(
            self.avg_metrics.metrics[GpuMetricEnum.GPU_FREE_MEM]
            / len(self.node_metrics),
            2,
        )
        self.avg_metrics.metrics[GpuMetricEnum.GPU_USED_MEM] = round(
            self.avg_metrics.metrics[GpuMetricEnum.GPU_USED_MEM]
            / len(self.node_metrics),
            2,
        )
        self.avg_metrics.metrics[GpuMetricEnum.GPU_UTIL] = round(
            self.avg_metrics.metrics[GpuMetricEnum.GPU_UTIL]
            / len(self.node_metrics),
            2,
        )
        self.avg_metrics.metrics[GpuMetricEnum.GPU_SM_UTIL] = round(
            self.avg_metrics.metrics[GpuMetricEnum.GPU_SM_UTIL]
            / len(self.node_metrics),
            2,
        )
        self.avg_metrics.metrics[GpuMetricEnum.GPU_TENSOR_UTIL] = round(
            self.avg_metrics.metrics[GpuMetricEnum.GPU_TENSOR_UTIL]
            / len(self.node_metrics),
            2,
        )


class NpuNodeMetric(XpuNodeMetric):
    """
    Metrics of all NPUs in a single node

    """

    def __init__(self):
        super().__init__()
        self.node_metrics: Dict[int, NpuMetric] = {}
        self.avg_metrics = NpuMetric()

    def get_avg_metric(self, metric):
        return self.avg_metrics.get_metric(metric)

    def get_node_metrics(self, metric):
        metrics = []
        for v in self.node_metrics.values():
            metrics.append(v.get_metric(metric))
        return metrics

    def update_avg_metrics(self):
        for _, metric in self.node_metrics.items():
            self.avg_metrics.metrics[
                NpuMetricEnum.NPU_TOTAL_MEM
            ] += metric.get_metric(NpuMetricEnum.NPU_TOTAL_MEM)
            self.avg_metrics.metrics[
                NpuMetricEnum.NPU_USED_MEM
            ] += metric.get_metric(NpuMetricEnum.NPU_USED_MEM)
            self.avg_metrics.metrics[
                NpuMetricEnum.NPU_UTIL
            ] += metric.get_metric(NpuMetricEnum.NPU_UTIL)

        self.avg_metrics.metrics[NpuMetricEnum.NPU_TOTAL_MEM] = round(
            self.avg_metrics.metrics[NpuMetricEnum.NPU_TOTAL_MEM]
            / len(self.node_metrics),
            2,
        )
        self.avg_metrics.metrics[NpuMetricEnum.NPU_USED_MEM] = round(
            self.avg_metrics.metrics[NpuMetricEnum.NPU_USED_MEM]
            / len(self.node_metrics),
            2,
        )
        self.avg_metrics.metrics[NpuMetricEnum.NPU_UTIL] = round(
            self.avg_metrics.metrics[NpuMetricEnum.NPU_UTIL]
            / len(self.node_metrics),
            2,
        )
