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

import copy
import http.client as http_client
import logging
import os
import time
import unittest
from datetime import datetime
from unittest.mock import patch

import requests

from dlrover.python.common.constants import (
    Accelerators,
    GpuMetricEnum,
    NpuMetricEnum,
)
from dlrover.python.common.metric.context import (
    JobMetricContext,
    get_job_metric_context,
)
from dlrover.python.common.metric.metric import (
    GpuMetric,
    GpuNodeMetric,
    NpuMetric,
    NpuNodeMetric,
)
from dlrover.python.common.metric.monitor import (
    GpuMetricMonitor,
    NpuMetricMonitor,
    SimpleMetricMonitor,
)

_metric_context = JobMetricContext.singleton_instance()


class MetricContextTests(unittest.TestCase):
    def test_gpu_metric(self):
        gmetric = GpuMetric(
            gpu_free_mem=50,
            gpu_used_mem=30,
            gpu_util=95,
            gpu_temperature=60,
            gpu_sm_util=0.05,
            gpu_tensor_util=0.1,
        )
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_FREE_MEM), 50)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_USED_MEM), 30)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_UTIL), 95)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_TEMP), 60)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_SM_UTIL), 0.05)
        self.assertEqual(
            gmetric.get_metric(GpuMetricEnum.GPU_TENSOR_UTIL), 0.1
        )

        gmetric.set_metric(GpuMetricEnum.GPU_FREE_MEM, 10)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_FREE_MEM), 10)
        gmetric.set_metric(GpuMetricEnum.GPU_USED_MEM, 70)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_USED_MEM), 70)
        gmetric.set_metric(GpuMetricEnum.GPU_UTIL, 10)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_UTIL), 10)
        gmetric.set_metric(GpuMetricEnum.GPU_TEMP, 80)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_TEMP), 80)
        gmetric.set_metric(GpuMetricEnum.GPU_SM_UTIL, 0.25)
        self.assertEqual(gmetric.get_metric(GpuMetricEnum.GPU_SM_UTIL), 0.25)
        gmetric.set_metric(GpuMetricEnum.GPU_TENSOR_UTIL, 0.3)
        self.assertEqual(
            gmetric.get_metric(GpuMetricEnum.GPU_TENSOR_UTIL), 0.3
        )

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
            npu_rx=1500,
        )
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_TOTAL_MEM), 80)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_USED_MEM), 75)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_UTIL), 95)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_TEMP), 70)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_HEALTH_STATE), 1)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_LINK_STATE), 0)
        self.assertEqual(
            gmetric.get_metric(NpuMetricEnum.NPU_OPTICAL_STATE), 0
        )
        self.assertEqual(
            gmetric.get_metric(NpuMetricEnum.NPU_NETWORK_STATE), 0
        )
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_RDMA_TX), 2000)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_RDMA_RX), 1500)

        gmetric.set_metric(NpuMetricEnum.NPU_TOTAL_MEM, 80)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_TOTAL_MEM), 80)
        gmetric.set_metric(NpuMetricEnum.NPU_USED_MEM, 20)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_USED_MEM), 20)
        gmetric.set_metric(NpuMetricEnum.NPU_UTIL, 2)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_UTIL), 2)
        gmetric.set_metric(NpuMetricEnum.NPU_TEMP, 30)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_TEMP), 30)
        gmetric.set_metric(NpuMetricEnum.NPU_HEALTH_STATE, 0)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_HEALTH_STATE), 0)
        gmetric.set_metric(NpuMetricEnum.NPU_LINK_STATE, 1)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_LINK_STATE), 1)
        gmetric.set_metric(NpuMetricEnum.NPU_OPTICAL_STATE, 1)
        self.assertEqual(
            gmetric.get_metric(NpuMetricEnum.NPU_OPTICAL_STATE), 1
        )
        gmetric.set_metric(NpuMetricEnum.NPU_NETWORK_STATE, 1)
        self.assertEqual(
            gmetric.get_metric(NpuMetricEnum.NPU_NETWORK_STATE), 1
        )
        gmetric.set_metric(NpuMetricEnum.NPU_RDMA_TX, 100)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_RDMA_TX), 100)
        gmetric.set_metric(NpuMetricEnum.NPU_RDMA_RX, 100)
        self.assertEqual(gmetric.get_metric(NpuMetricEnum.NPU_RDMA_RX), 100)

    def test_node_gpu_metric(self):
        worker1 = GpuNodeMetric()
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

        worker1.update_avg_metrics()
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricEnum.GPU_FREE_MEM], 50
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricEnum.GPU_USED_MEM], 30
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricEnum.GPU_UTIL], 90
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricEnum.GPU_TEMP], 0
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricEnum.GPU_SM_UTIL], 0.02
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[GpuMetricEnum.GPU_TENSOR_UTIL], 0.23
        )

    def test_node_npu_metric(self):
        worker1 = NpuNodeMetric()
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
            npu_rx=1500,
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
            npu_rx=1500,
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
            npu_rx=1500,
        )

        worker1.update_avg_metrics()
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricEnum.NPU_TOTAL_MEM], 80
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricEnum.NPU_USED_MEM], 70
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricEnum.NPU_UTIL], 97.33
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricEnum.NPU_TEMP], 0
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricEnum.NPU_RDMA_RX], 0.0
        )
        self.assertEqual(
            worker1.avg_metrics.metrics[NpuMetricEnum.NPU_RDMA_TX], 0.0
        )

    def test_job_metric(self):
        jctx = get_job_metric_context()
        nmetric1 = GpuNodeMetric()
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

        nmetric2 = GpuNodeMetric()
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
                "worker-2": copy.deepcopy(nmetric1),
            },
        )
        time.sleep(1)
        key2 = int(time.time())
        jctx.add_node_metrics(
            key2,
            {
                "worker-1": copy.deepcopy(nmetric2),
                "worker-2": copy.deepcopy(nmetric2),
            },
        )
        self.assertEqual(jctx.get_earliest_node_metrics()[0], key1)
        self.assertEqual(jctx.get_latest_node_metrics()[0], key2)
        self.assertEqual(jctx.size(), 2)

        jctx.max_metric_records = 2
        time.sleep(1)
        key3 = int(time.time())
        jctx.add_node_metrics(
            key3,
            {
                "worker-1": copy.deepcopy(nmetric1),
                "worker-2": copy.deepcopy(nmetric1),
            },
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


class MockMetricResponse(object):
    def __init__(self, json_data, status_code):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


def mock_requests_conn_exception(*args, **kwargs):
    raise requests.exceptions.ConnectionError


def mock_requests_http_exception(*args, **kwargs):
    raise requests.exceptions.HTTPError


def mock_requests_timeout_exception(*args, **kwargs):
    raise requests.exceptions.Timeout


def mock_requests_request_exception(*args, **kwargs):
    raise requests.exceptions.RequestException


def mock_npu_pod_metric_request(*args, **kwargs):
    npu_pod_sample = {
        "success": True,
        "message": "success",
        "data": [
            [
                {
                    "tags": {
                        "pod_name": "dlrover-testjob-worker-100",
                        "id": "0",
                        "job": "dlrover-testjob",
                        "gpuType": "ascend.NPU",
                    },
                    "dataMapByTime": {"1732251300000": {"count": "95"}},
                },
                {
                    "tags": {
                        "pod_name": "dlrover-testjob-worker-100",
                        "id": "1",
                        "job": "dlrover-testjob",
                        "gpuType": "ascend.NPU",
                    },
                    "dataMapByTime": {"1732251300000": {"count": "98"}},
                },
                {
                    "tags": {
                        "pod_name": "dlrover-testjob-worker-100",
                        "id": "2",
                        "job": "dlrover-testjob",
                        "gpuType": "ascend.NPU",
                    },
                    "dataMapByTime": {"1732251300000": {"count": "99"}},
                },
                {
                    "tags": {
                        "pod_name": "dlrover-testjob-worker-100",
                        "id": "3",
                        "job": "dlrover-testjob",
                        "gpuType": "ascend.NPU",
                    },
                    "dataMapByTime": {"1732251300000": {"count": "99"}},
                },
            ]
        ],
    }
    return MockMetricResponse(npu_pod_sample, 200)


def mock_npu_job_metric_request(*args, **kwargs):
    npu_job_sample = {
        "success": True,
        "message": "success",
        "data": [
            [
                {
                    "tags": {
                        "pod_name": "dlrover-testjob-worker-100",
                        "id": "0",
                        "job": "dlrover-testjob",
                        "gpuType": "ascend.NPU",
                    },
                    "dataMapByTime": {"1732251300000": {"count": "95"}},
                },
                {
                    "tags": {
                        "pod_name": "dlrover-testjob-worker-100",
                        "id": "1",
                        "job": "dlrover-testjob",
                        "gpuType": "ascend.NPU",
                    },
                    "dataMapByTime": {"1732251300000": {"count": "98"}},
                },
                {
                    "tags": {
                        "pod_name": "dlrover-testjob-worker-101",
                        "id": "0",
                        "job": "dlrover-testjob",
                        "gpuType": "ascend.NPU",
                    },
                    "dataMapByTime": {"1732251300000": {"count": "99"}},
                },
                {
                    "tags": {
                        "pod_name": "dlrover-testjob-worker-101",
                        "id": "1",
                        "job": "dlrover-testjob",
                        "gpuType": "ascend.NPU",
                    },
                    "dataMapByTime": {"1732251300000": {"count": "99"}},
                },
            ]
        ],
    }
    return MockMetricResponse(npu_job_sample, 200)


def mock_gpu_pod_metric_request(*args, **kwargs):
    gpu_pod_sample = {
        "success": True,
        "message": "success",
        "data": [
            [
                {
                    "tags": {
                        "pod": "dlrover-testjob-worker-100",
                        "gpu": "0",
                        "job": "dlrover-testjob",
                        "gpuType": "nvidia.GPU",
                    },
                    "dataMapByTime": {"1732442400000": {"count": "95"}},
                },
                {
                    "tags": {
                        "pod": "dlrover-testjob-worker-100",
                        "gpu": "1",
                        "job": "dlrover-testjob",
                        "gpuType": "nvidia.GPU",
                    },
                    "dataMapByTime": {"1732442400000": {"count": "98"}},
                },
                {
                    "tags": {
                        "pod": "dlrover-testjob-worker-100",
                        "gpu": "2",
                        "job": "dlrover-testjob",
                        "gpuType": "nvidia.GPU",
                    },
                    "dataMapByTime": {"1732442400000": {"count": "99"}},
                },
                {
                    "tags": {
                        "pod": "dlrover-testjob-worker-100",
                        "gpu": "3",
                        "job": "dlrover-testjob",
                        "gpuType": "nvidia.GPU",
                    },
                    "dataMapByTime": {"1732442400000": {"count": "99"}},
                },
            ]
        ],
    }
    return MockMetricResponse(gpu_pod_sample, 200)


def mock_gpu_job_metric_request(*args, **kwargs):
    gpu_job_sample = {
        "success": True,
        "message": "success",
        "data": [
            [
                {
                    "tags": {
                        "pod": "dlrover-testjob-worker-100",
                        "gpu": "0",
                        "job": "dlrover-testjob",
                        "gpuType": "nvidia.GPU",
                    },
                    "dataMapByTime": {"1732442400000": {"count": "95"}},
                },
                {
                    "tags": {
                        "pod": "dlrover-testjob-worker-100",
                        "gpu": "1",
                        "job": "dlrover-testjob",
                        "gpuType": "nvidia.GPU",
                    },
                    "dataMapByTime": {"1732442400000": {"count": "98"}},
                },
                {
                    "tags": {
                        "pod": "dlrover-testjob-worker-101",
                        "gpu": "0",
                        "job": "dlrover-testjob",
                        "gpuType": "nvidia.GPU",
                    },
                    "dataMapByTime": {"1732442400000": {"count": "99"}},
                },
                {
                    "tags": {
                        "pod": "dlrover-testjob-worker-101",
                        "gpu": "1",
                        "job": "dlrover-testjob",
                        "gpuType": "nvidia.GPU",
                    },
                    "dataMapByTime": {"1732442400000": {"count": "99"}},
                },
            ]
        ],
    }
    return MockMetricResponse(gpu_job_sample, 200)


def mock_gpu_job_metric_exception(*args, **kwargs):
    pass


class MetricMonitorTests(unittest.TestCase):
    def setUp(self):
        os.environ["DLROVER_METRIC_URL"] = "https://metric.mock.dlrover.org"
        os.environ["DLROVER_METRIC_TOKEN"] = "test"
        http_client.HTTPConnection.debuglevel = 1
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        _metric_context.clear_node_metrics()

    def tearDown(self):
        _metric_context.clear_node_metrics()

    def test_query_exception(self):
        job_name = "dlrover-testjob"
        metric_type = NpuMetricEnum.NPU_UTIL

        mon = SimpleMetricMonitor(job_name, [metric_type])

        start = datetime.strptime(
            "2024-11-22 4:55:00", "%Y-%m-%d %H:%M:%S"
        ).timestamp()
        end = datetime.strptime(
            "2024-11-22 4:55:45", "%Y-%m-%d %H:%M:%S"
        ).timestamp()

        with patch("requests.post", side_effect=mock_requests_conn_exception):
            rsp = mon.query_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                is_gpu=False,
            )
            self.assertEqual(rsp, None)

        with patch("requests.post", side_effect=mock_requests_http_exception):
            rsp = mon.query_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                is_gpu=False,
            )
            self.assertEqual(rsp, None)

        with patch(
            "requests.post", side_effect=mock_requests_timeout_exception
        ):
            rsp = mon.query_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                is_gpu=False,
            )
            self.assertEqual(rsp, None)

        with patch(
            "requests.post", side_effect=mock_requests_request_exception
        ):
            rsp = mon.query_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                is_gpu=False,
            )
            self.assertEqual(rsp, None)

    def test_query_npu_job_metrics(self):
        with patch("requests.post", side_effect=mock_npu_job_metric_request):
            job_name = "dlrover-testjob"
            metric_type = NpuMetricEnum.NPU_UTIL

            mon = SimpleMetricMonitor(job_name, [metric_type])

            start = datetime.strptime(
                "2024-11-22 4:55:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            end = datetime.strptime(
                "2024-11-22 4:55:45", "%Y-%m-%d %H:%M:%S"
            ).timestamp()

            rsp = mon.query_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                is_gpu=False,
            )
            self.assertTrue(rsp["success"])

    def test_query_npu_pod_metrics(self):
        with patch("requests.post", side_effect=mock_npu_pod_metric_request):
            job_name = "dlrover-testjob"
            pod_name = "dlrover-testjob-worker-100"
            metric_type = NpuMetricEnum.NPU_UTIL

            mon = SimpleMetricMonitor(job_name, [metric_type])

            start = int(
                datetime.strptime(
                    "2024-11-22 4:55:00", "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            )
            end = int(
                datetime.strptime(
                    "2024-11-22 4:56:00", "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            )

            rsp = mon.query_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                is_gpu=False,
                pod_name=pod_name,
            )
            self.assertTrue(rsp["success"])

    def test_query_gpu_job_metrics(self):
        with patch("requests.post", side_effect=mock_gpu_job_metric_request):
            job_name = "dlrover-testjob"
            metric_type = GpuMetricEnum.GPU_UTIL

            mon = SimpleMetricMonitor(job_name, [metric_type])

            start = datetime.strptime(
                "2024-11-24 10:00:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            end = datetime.strptime(
                "2024-11-24 10:05:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()

            rsp = mon.query_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                is_gpu=True,
            )
            self.assertTrue(rsp["success"])

    def test_query_gpu_pod_metrics(self):
        with patch("requests.post", side_effect=mock_gpu_pod_metric_request):
            job_name = "dlrover-testjob"
            pod_name = "dlrover-testjob-worker-100"
            metric_type = GpuMetricEnum.GPU_UTIL

            mon = SimpleMetricMonitor(job_name, [metric_type])

            start = int(
                datetime.strptime(
                    "2024-11-24 10:00:00", "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            )
            end = int(
                datetime.strptime(
                    "2024-11-24 10:03:00", "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            )

            rsp = mon.query_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                is_gpu=True,
                pod_name=pod_name,
            )
            self.assertTrue(rsp["success"])

    def test_collect_npu_job_metrics(self):
        with patch("requests.post", side_effect=mock_npu_job_metric_request):
            job_name = "dlrover-testjob"
            metric_type = NpuMetricEnum.NPU_UTIL

            mon = NpuMetricMonitor(job_name, [metric_type])

            start = datetime.strptime(
                "2024-11-22 4:55:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            end = datetime.strptime(
                "2024-11-22 4:55:45", "%Y-%m-%d %H:%M:%S"
            ).timestamp()

            job_metric = mon.collect_job_metrics(
                job_name,
                metric_type,
                start,
                end,
            )

            self.assertIsNotNone(job_metric)

            for pod, node_metric in job_metric.items():
                if pod == "dlrover-testjob-worker-100":
                    # node_metric.update_avg_metrics()
                    self.assertEqual(
                        node_metric.avg_metrics.get_metric(
                            NpuMetricEnum.NPU_UTIL
                        ),
                        96.5,
                    )
                elif pod == "dlrover-testjob-worker-101":
                    # node_metric.update_avg_metrics()
                    self.assertEqual(
                        node_metric.avg_metrics.get_metric(
                            NpuMetricEnum.NPU_UTIL
                        ),
                        99.0,
                    )

    def test_collect_npu_pod_metrics(self):
        with patch("requests.post", side_effect=mock_npu_pod_metric_request):
            job_name = "dlrover-testjob"
            pod_name = "dlrover-testjob-worker-100"
            metric_type = NpuMetricEnum.NPU_UTIL

            mon = NpuMetricMonitor(job_name, [metric_type])

            start = datetime.strptime(
                "2024-11-22 4:55:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            end = datetime.strptime(
                "2024-11-22 4:55:45", "%Y-%m-%d %H:%M:%S"
            ).timestamp()

            pod_metric = mon.collect_job_metrics(
                job_name,
                metric_type,
                start,
                end,
                pod_name=pod_name,
            )

            self.assertIsNotNone(pod_metric)

            for pod, node_metric in pod_metric.items():
                # node_metric.update_avg_metrics()
                self.assertEqual(
                    node_metric.avg_metrics.get_metric(NpuMetricEnum.NPU_UTIL),
                    97.75,
                )

    def test_collect_gpu_job_metrics(self):
        with patch("requests.post", side_effect=mock_gpu_job_metric_request):
            job_name = "dlrover-testjob"
            metric_type = GpuMetricEnum.GPU_UTIL

            mon = GpuMetricMonitor(job_name, [metric_type])

            start = datetime.strptime(
                "2024-11-24 10:00:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            end = datetime.strptime(
                "2024-11-24 10:05:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()

            job_metric = mon.collect_job_metrics(
                job_name,
                metric_type,
                start,
                end,
            )

            self.assertIsNotNone(job_metric)

            for pod, node_metric in job_metric.items():
                if pod == "dlrover-testjob-worker-100":
                    node_metric.update_avg_metrics()
                    self.assertEqual(
                        node_metric.avg_metrics.get_metric(
                            GpuMetricEnum.GPU_UTIL
                        ),
                        96.5,
                    )
                elif pod == "dlrover-testjob-worker-101":
                    node_metric.update_avg_metrics()
                    self.assertEqual(
                        node_metric.avg_metrics.get_metric(
                            GpuMetricEnum.GPU_UTIL
                        ),
                        99.0,
                    )

    def test_collect_gpu_pod_metrics(self):
        with patch("requests.post", side_effect=mock_gpu_pod_metric_request):
            job_name = "dlrover-testjob"
            pod_name = "dlrover-testjob-worker-100"
            metric_type = GpuMetricEnum.GPU_UTIL

            mon = GpuMetricMonitor(job_name, [metric_type])

            start = datetime.strptime(
                "2024-11-24 10:00:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()
            end = datetime.strptime(
                "2024-11-24 10:05:00", "%Y-%m-%d %H:%M:%S"
            ).timestamp()

            job_metric = mon.collect_job_metrics(
                job_name, metric_type, start, end, pod_name=pod_name
            )

            self.assertIsNotNone(job_metric)

            for pod, node_metric in job_metric.items():
                node_metric.update_avg_metrics()
                self.assertEqual(
                    node_metric.avg_metrics.get_metric(GpuMetricEnum.GPU_UTIL),
                    97.75,
                )


def mock_gpu_monitor_collect(*args, **kwargs):
    if "metrics" in kwargs:
        job_metrics = kwargs["metrics"]
        metric = GpuNodeMetric()
        for i in range(8):
            metric.node_metrics[i] = GpuMetric()
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_FREE_MEM, 0)
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_USED_MEM, 80)
            metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_UTIL, 99.5)
            metric.node_metrics[i].set_metric(
                GpuMetricEnum.GPU_TENSOR_UTIL, 30.5
            )
        metric.update_avg_metrics()
        job_metrics["worker-1"] = copy.deepcopy(metric)
        job_metrics["worker-2"] = copy.deepcopy(metric)
        job_metrics["worker-3"] = copy.deepcopy(metric)
        job_metrics["worker-4"] = copy.deepcopy(metric)

        return job_metrics


def mock_npu_monitor_collect(*args, **kwargs):
    if "metrics" in kwargs:
        job_metrics = kwargs["metrics"]
        metric = NpuNodeMetric()
        for i in range(8):
            metric.node_metrics[i] = NpuMetric()
            metric.node_metrics[i].set_metric(NpuMetricEnum.NPU_UTIL, 99.5)
        metric.update_avg_metrics()
        job_metrics["worker-1"] = copy.deepcopy(metric)
        job_metrics["worker-2"] = copy.deepcopy(metric)
        job_metrics["worker-3"] = copy.deepcopy(metric)
        job_metrics["worker-4"] = copy.deepcopy(metric)

        return job_metrics


class GpuMetricMonitorTest(unittest.TestCase):
    def setUp(self):
        self.url = os.getenv("DLROVER_METRIC_URL", "")
        self.token = os.getenv("DLROVER_METRIC_TOKEN", "")
        self.job_name = os.getenv("DLROVER_JOB_NAME", "")
        self.xpu = Accelerators.NVIDIA_GPU
        http_client.HTTPConnection.debuglevel = 1
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)

    def tearDown(self):
        _metric_context.clear_node_metrics()

    def test_gpu_collector(self):
        with patch(
            "dlrover.python.common.metric.monitor.GpuMetricMonitor.collect_job_metrics",  # noqa
            side_effect=mock_gpu_monitor_collect,
        ):
            mon = GpuMetricMonitor(
                job_name=self.job_name,
                metrics=[
                    GpuMetricEnum.GPU_TENSOR_UTIL,
                    GpuMetricEnum.GPU_UTIL,
                ],
            )
            mon.start(interval=1)
            time.sleep(2)
            mon.stop()

            _metric_context.log_job_metric(GpuMetricEnum.GPU_TENSOR_UTIL)
            _metric_context.log_job_metric(GpuMetricEnum.GPU_UTIL)

            self.assertEqual(_metric_context.size(), 1)

            self.assertEqual(
                list(
                    _metric_context.backtrace_avg_metrics(
                        GpuMetricEnum.GPU_TENSOR_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [30.5],
            )
            self.assertEqual(
                list(
                    _metric_context.backtrace_node_metrics(
                        GpuMetricEnum.GPU_TENSOR_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [[30.5, 30.5, 30.5, 30.5]],
            )
            self.assertEqual(
                list(
                    _metric_context.backtrace_rank_metrics(
                        GpuMetricEnum.GPU_TENSOR_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [
                    [
                        [30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5],
                        [30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5],
                        [30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5],
                        [30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5, 30.5],
                    ]
                ],
            )

            self.assertEqual(
                list(
                    _metric_context.backtrace_avg_metrics(
                        GpuMetricEnum.GPU_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [99.5],
            )
            self.assertEqual(
                list(
                    _metric_context.backtrace_node_metrics(
                        GpuMetricEnum.GPU_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [[99.5, 99.5, 99.5, 99.5]],
            )
            self.assertEqual(
                list(
                    _metric_context.backtrace_rank_metrics(
                        GpuMetricEnum.GPU_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [
                    [
                        [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                        [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                        [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                        [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                    ]
                ],
            )

            _metric_context.clear_node_metrics()
            self.assertEqual(_metric_context.size(), 0)


class NpuMetricMonitorTest(unittest.TestCase):
    def setUp(self):
        self.url = os.getenv("DLROVER_METRIC_URL", "")
        self.token = os.getenv("DLROVER_METRIC_TOKEN", "")
        self.job_name = os.getenv("DLROVER_JOB_NAME", "")
        self.xpu = Accelerators.ASCEND_NPU
        http_client.HTTPConnection.debuglevel = 1
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)

    def tearDown(self):
        _metric_context.clear_node_metrics()

    def test_npu_collector(self):
        with patch(
            "dlrover.python.common.metric.monitor.NpuMetricMonitor.collect_job_metrics",  # noqa
            side_effect=mock_npu_monitor_collect,
        ):
            mon = NpuMetricMonitor(
                job_name=self.job_name,
                metrics=[
                    NpuMetricEnum.NPU_UTIL,
                ],
            )
            mon.start(interval=1)
            time.sleep(2)
            mon.stop()

            _metric_context.log_job_metric(NpuMetricEnum.NPU_UTIL)

            self.assertEqual(_metric_context.size(), 1)
            self.assertEqual(
                list(
                    _metric_context.backtrace_avg_metrics(
                        NpuMetricEnum.NPU_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [99.5],
            )
            self.assertEqual(
                list(
                    _metric_context.backtrace_node_metrics(
                        NpuMetricEnum.NPU_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [[99.5, 99.5, 99.5, 99.5]],
            )
            self.assertEqual(
                list(
                    _metric_context.backtrace_rank_metrics(
                        NpuMetricEnum.NPU_UTIL,
                        _metric_context.max_metric_records,
                    ).values()
                ),
                [
                    [
                        [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                        [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                        [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                        [99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5, 99.5],
                    ]
                ],
            )

            _metric_context.clear_node_metrics()
            self.assertEqual(_metric_context.size(), 0)
