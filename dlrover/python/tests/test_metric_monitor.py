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

import unittest

import os
import threading
import time
import copy
import requests
import json

from datetime import datetime
from collections import OrderedDict
from typing import (
    Dict,
    Any,
    List,
    Optional,
)

from dlrover.python.common.constants import (
    GpuMetricType,
    NpuMetricType,
)

from dlrover.python.common.serialize import JsonSerializable
from dlrover.python.common.singleton import Singleton
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger

from dlrover.python.master.monitor.metric_context import (
    JobMetricContext,
    NodeGpuMetric,
    NodeNpuMetric,
    GpuMetric,
    NpuMetric,
    get_job_metric_context,
)

from dlrover.python.master.monitor.metric_monitor import (
    GpuMetricMonitor,
    NpuMetricMonitor,
)

_dlrover_context = Context.singleton_instance()

class MetricMonitorTests(unittest.TestCase):
    def test_gpu_metric(self):
        gmetric = GpuMetric(
            gpu_free_mem=50,
            gpu_used_mem=30,
            gpu_util=95,
            gpu_temperature=60,
            gpu_sm_util=0.05,
            gpu_tensor_util=0.1,
        )
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_FREE_MEM), 50)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_USED_MEM), 30)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_UTIL), 95)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_TEMP), 60)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_SM_UTIL), 0.05)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_TENSOR_UTIL), 0.1)

        gmetric.set_metric(GpuMetricType.GPU_FREE_MEM, 10)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_FREE_MEM), 10)
        gmetric.set_metric(GpuMetricType.GPU_USED_MEM, 70)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_USED_MEM), 70)
        gmetric.set_metric(GpuMetricType.GPU_UTIL, 10)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_UTIL), 10)
        gmetric.set_metric(GpuMetricType.GPU_TEMP, 80)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_TEMP), 80)
        gmetric.set_metric(GpuMetricType.GPU_SM_UTIL, 0.25)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_SM_UTIL), 0.25)
        gmetric.set_metric(GpuMetricType.GPU_TENSOR_UTIL, 0.3)
        self.assertEqual(gmetric.get_metric(GpuMetricType.GPU_TENSOR_UTIL), 0.3)

    def test_npu_metric(self):
        gmetric = NpuMetric(
            npu_total_mem=80,
            npu_used_mem=75,
            npu_util=95,
            npu_temperature=70,
            npu_health_state=1,
            npu_link_state=0,
            npu_optical_state=0,
            npu_network_state=0,
            npu_tx=2000,
            npu_rx=1500
        )
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_TOTAL_MEM), 80)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_USED_MEM), 75)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_UTIL), 95)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_TEMP), 70)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_HEALTH_STATE), 1)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_LINK_STATE), 0)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_OPTICAL_STATE), 0)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_NETWORK_STATE), 0)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_RDMA_TX), 2000)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_RDMA_RX), 1500)

        gmetric.set_metric(NpuMetricType.NPU_TOTAL_MEM, 80)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_TOTAL_MEM), 80)
        gmetric.set_metric(NpuMetricType.NPU_USED_MEM, 20)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_USED_MEM), 20)
        gmetric.set_metric(NpuMetricType.NPU_UTIL, 2)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_UTIL), 2)
        gmetric.set_metric(NpuMetricType.NPU_TEMP, 30)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_TEMP), 30)
        gmetric.set_metric(NpuMetricType.NPU_HEALTH_STATE, 0)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_HEALTH_STATE), 0)
        gmetric.set_metric(NpuMetricType.NPU_LINK_STATE, 1)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_LINK_STATE), 1)
        gmetric.set_metric(NpuMetricType.NPU_OPTICAL_STATE, 1)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_OPTICAL_STATE), 1)
        gmetric.set_metric(NpuMetricType.NPU_NETWORK_STATE, 1)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_NETWORK_STATE), 1)
        gmetric.set_metric(NpuMetricType.NPU_RDMA_TX, 100)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_RDMA_TX), 100)
        gmetric.set_metric(NpuMetricType.NPU_RDMA_RX, 100)
        self.assertEqual(gmetric.get_metric(NpuMetricType.NPU_RDMA_RX), 100)

    def test_node_gpu_metric(self):
        worker1 = NodeGpuMetric()
        worker1.node_metrics[0] = GpuMetric(
            gpu_free_mem=50,
            gpu_used_mem=30,
            gpu_util=95,
            gpu_temperature=60,
            gpu_sm_util=0.05,
            gpu_tensor_util=0.1,
        )
        worker1.node_metrics[1] = GpuMetric(
            gpu_free_mem=60,
            gpu_used_mem=20,
            gpu_util=90,
            gpu_temperature=60,
            gpu_sm_util=0.01,
            gpu_tensor_util=0.2,
        )
        worker1.node_metrics[2] = GpuMetric(
            gpu_free_mem=40,
            gpu_used_mem=40,
            gpu_util=85,
            gpu_temperature=60,
            gpu_sm_util=0.00,
            gpu_tensor_util=0.4,
        )

        self.assertEqual(worker1.avg_metrics, None)
        worker1.update_avg_metrics()
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricType.GPU_FREE_MEM], 50
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricType.GPU_USED_MEM], 30
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricType.GPU_UTIL], 90
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricType.GPU_TEMP], 0
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricType.GPU_SM_UTIL], 0.02
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricType.GPU_TENSOR_UTIL], 0.23
        )

    def test_node_npu_metric(self):
        worker1 = NodeNpuMetric()
        worker1.node_metrics[0] = NpuMetric(
            npu_total_mem=80,
            npu_used_mem=75,
            npu_util=95,
            npu_temperature=70,
            npu_health_state=1,
            npu_link_state=0,
            npu_optical_state=0,
            npu_network_state=0,
            npu_tx=2000,
            npu_rx=1500
        )
        worker1.node_metrics[1] = NpuMetric(
            npu_total_mem=80,
            npu_used_mem=70,
            npu_util=99,
            npu_temperature=70,
            npu_health_state=1,
            npu_link_state=0,
            npu_optical_state=0,
            npu_network_state=0,
            npu_tx=2000,
            npu_rx=1500
        )
        worker1.node_metrics[2] = NpuMetric(
            npu_total_mem=80,
            npu_used_mem=65,
            npu_util=98,
            npu_temperature=70,
            npu_health_state=1,
            npu_link_state=0,
            npu_optical_state=0,
            npu_network_state=0,
            npu_tx=2000,
            npu_rx=1500
        )

        self.assertEqual(worker1.avg_metrics, None)
        worker1.update_avg_metrics()
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricType.NPU_TOTAL_MEM], 80
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricType.NPU_USED_MEM], 70
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricType.NPU_UTIL], 97.33
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricType.NPU_TEMP], 0
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricType.NPU_RDMA_RX], 0.0
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricType.NPU_RDMA_TX], 0.0
        )


    def test_gpu_job_metric(self):
        jctx = get_job_metric_context()
        nmetric1 = NodeGpuMetric()
        nmetric1.node_metrics[0] = GpuMetric(
            gpu_free_mem=50,
            gpu_used_mem=30,
            gpu_util=95,
            gpu_temperature=60,
            gpu_sm_util=0.05,
            gpu_tensor_util=0.1,
        )
        nmetric1.node_metrics[1] = GpuMetric(
            gpu_free_mem=60,
            gpu_used_mem=20,
            gpu_util=90,
            gpu_temperature=60,
            gpu_sm_util=0.01,
            gpu_tensor_util=0.2,
        )
        nmetric1.node_metrics[2] = GpuMetric(
            gpu_free_mem=40,
            gpu_used_mem=40,
            gpu_util=85,
            gpu_temperature=60,
            gpu_sm_util=0.00,
            gpu_tensor_util=0.4,
        )

        nmetric2 = NodeGpuMetric()
        nmetric2.node_metrics[0] = GpuMetric(
            gpu_free_mem=10,
            gpu_used_mem=70,
            gpu_util=45,
            gpu_temperature=60,
            gpu_sm_util=0.05,
            gpu_tensor_util=0.1,
        )
        nmetric2.node_metrics[1] = GpuMetric(
            gpu_free_mem=20,
            gpu_used_mem=60,
            gpu_util=55,
            gpu_temperature=60,
            gpu_sm_util=0.01,
            gpu_tensor_util=0.2,
        )
        nmetric2.node_metrics[2] = GpuMetric(
            gpu_free_mem=30,
            gpu_used_mem=50,
            gpu_util=35,
            gpu_temperature=60,
            gpu_sm_util=0.00,
            gpu_tensor_util=0.4,
        )

        key1 = int(time.time())
        jctx.add_node_metrics(
            key1,
            {
                "worker-1": copy.deepcopy(nmetric1),
                "worker-2": copy.deepcopy(nmetric1)
            }
        )
        time.sleep(1)
        key2 = int(time.time())
        jctx.add_node_metrics(
            key2,
            {
                "worker-1": copy.deepcopy(nmetric2),
                "worker-2": copy.deepcopy(nmetric2)
            }
        )
        self.assertEqual(
            jctx.get_earliest_node_metrics()[0],
            key1
        )
        self.assertEqual(
            jctx.get_latest_node_metrics()[0],
            key2
        )
        self.assertEqual(jctx.size(), 2)

        jctx.max_metric_records = 2
        time.sleep(1)
        key3 = int(time.time())
        jctx.add_node_metrics(
            key3,
            {
                "worker-1": copy.deepcopy(nmetric1),
                "worker-2": copy.deepcopy(nmetric1)
            }
        )
        self.assertEqual(
            jctx.get_latest_node_metrics()[0],
            key3,
        )
        self.assertEqual(
            jctx.get_earliest_node_metrics()[0],
            key2,
        )
        self.assertEqual(jctx.size(), 2)

        self.assertEqual(len(jctx.get_node_metrics()), 2)
        jctx.clear_node_metrics()
        self.assertEqual(jctx.size(), 0)
        self.assertEqual(len(jctx.get_node_metrics()), 0)