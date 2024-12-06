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

import os
import traceback
from abc import ABCMeta, abstractmethod
from datetime import datetime

import requests
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

from dlrover.python.common.constants import GpuMetricEnum, NpuMetricEnum
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.metric.context import JobMetricContext
from dlrover.python.common.metric.metric import (
    GpuMetric,
    GpuNodeMetric,
    NpuMetric,
    NpuNodeMetric,
)

_dlrover_context = Context.singleton_instance()
_metric_context = JobMetricContext.singleton_instance()


class MetricMonitor(metaclass=ABCMeta):
    """
    metric monitor abstract api
    """

    def __init__(self):
        pass

    @abstractmethod
    def collect_job_metrics(
        self, job_name, metric_enum, start, end, pod_name=None, metrics=None
    ):
        """
        collect and update job metric

        Args:
            job_name: job name
            metric_enum: metric enum
            start: start timestamp
            end: end timestamp
            pod_name: pod name as a pod filter
            metrics: job metrics dict, with type as Dict[str, NodeXpuMetric]

        Returns:
            Dict with keys as pod name and
            values as list of metric value with local rank id as index

        """
        pass


class SimpleMetricMonitor(MetricMonitor):
    """
    implementation of metric monitor that uses http REST api to query metrics
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def build_request_headers(token):
        return {"apitoken": token, "Content-Type": "application/json"}

    @staticmethod
    def build_job_request_data(metric, job, start, end):
        condition = {
            "query": f"{metric}" + r"{job=" + f'"{job}"' + r"}",
            "start": start,
            "end": end,
            "step": 60000,
            "metricType": "stack",
            "tag": "PROD",
            "metrics": [],
            "stack": "antLLM",
            "tenantId": 1,
            "workspaceId": -1,
        }
        return [
            {
                "condition": condition,
                "outformat": "trend",
                "datasource": "pqlTrendQuery",
            }
        ]

    @staticmethod
    def build_npu_pod_request_data(metric, pod, start, end):
        condition = {
            "query": f"{metric}" + r"{pod_name=" + f'"{pod}"' + r"}",
            "start": start,
            "end": end,
            "step": 60000,
            "metricType": "stack",
            "tag": "PROD",
            "metrics": [],
            "stack": "antLLM",
            "tenantId": 1,
            "workspaceId": -1,
        }
        return [
            {
                "condition": condition,
                "outformat": "trend",
                "datasource": "pqlTrendQuery",
            }
        ]

    @staticmethod
    def build_gpu_pod_request_data(metric, pod, start, end):
        condition = {
            "query": f"{metric}" + r"{pod=" + f'"{pod}"' + r"}",
            "start": start,
            "end": end,
            "step": 60000,
            "metricType": "stack",
            "tag": "PROD",
            "metrics": [],
            "stack": "antLLM",
            "tenantId": 1,
            "workspaceId": -1,
        }
        return [
            {
                "condition": condition,
                "outformat": "trend",
                "datasource": "pqlTrendQuery",
            }
        ]

    @staticmethod
    def adjust_timestamp(start, end):
        """
        adjust (start, end) to into 1 minute, starting from 0 sec of a minute,
        and end with 59 sec of the minute

        Args:
            start: start timestamp
            end: end timestamp

        Returns:
        a new (start, end) with unit of milliseconds
        """
        dt_obj = datetime.fromtimestamp(start)
        start_str = "{}-{}-{} {}:{}:{}".format(
            dt_obj.year,
            dt_obj.month,
            dt_obj.day,
            dt_obj.hour,
            dt_obj.minute,
            "00",
        )
        end_str = "{}-{}-{} {}:{}:{}".format(
            dt_obj.year,
            dt_obj.month,
            dt_obj.day,
            dt_obj.hour,
            dt_obj.minute,
            "59",
        )
        start_obj = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end_obj = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        new_start = int(start_obj.timestamp())
        new_end = int(end_obj.timestamp())

        return new_start * 1000, new_end * 1000

    def query_job_metrics(
        self, job_name, metric_type, start, end, is_gpu=True, pod_name=None
    ):
        url = os.getenv("DLROVER_METRIC_URL", "")
        if url == "":
            logger.warning("No GPU metrics url defined")
            return None
        token = os.getenv("DLROVER_METRIC_TOKEN", "")
        if token == "":
            logger.warning("No GPU metrics token defined")
            return None

        try:
            start_time, end_time = self.adjust_timestamp(start, end)
            headers = self.build_request_headers(token)
            if pod_name is None:
                data = self.build_job_request_data(
                    metric_type, job_name, start_time, end_time
                )
            elif is_gpu is True:
                data = self.build_gpu_pod_request_data(
                    metric_type, pod_name, start_time, end_time
                )
            else:
                data = self.build_npu_pod_request_data(
                    metric_type, pod_name, start_time, end_time
                )
            resp = requests.post(url, headers=headers, json=data)

            if resp.status_code != requests.codes.ok:
                logger.warning(
                    f"Failed to query {job_name} {metric_type}: "
                    f"{resp.status_code}"
                )
                return None
            rsp = resp.json()
            if rsp["success"] is not True:
                logger.warning(
                    f"{job_name} {metric_type} response result failed"
                )
                return None

            return rsp

        except Timeout as e:
            logger.warning(f"Request timed out: {e}")
            return None
        except HTTPError as e:
            logger.warning(f"Request HTTP failed: {e}")
            return None
        except ConnectionError as e:
            logger.warning(f"Connection error: {e}")
            return None
        except RequestException as e:
            logger.warning(f"Unexpected Request exception: {e}")
            return None
        except KeyError as e:
            logger.warning(f"Key error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error: {e}")
            return None

    def collect_job_metrics(
        self, job_name, metric_type, start, end, pod_name=None, metrics=None
    ):
        pass


class GpuMetricMonitor(SimpleMetricMonitor):
    """
    metric monitor of nvidia GPU metrics
    """

    def __init__(self):
        super().__init__()

    def collect_job_metrics(
        self, job_name, metric_type, start, end, pod_name=None, metrics=None
    ):
        rsp = self.query_job_metrics(
            job_name, metric_type, start, end, is_gpu=True, pod_name=pod_name
        )
        if rsp is None:
            return None

        try:
            new_start, _ = self.adjust_timestamp(start, end)
            start_time = f"{int(new_start)}"
            job_metrics = {} if metrics is None else metrics

            for rank_metric in rsp["data"][0]:
                pod = rank_metric["tags"]["pod"]
                local_rank = int(rank_metric["tags"]["gpu"])
                if (
                    metric_type == GpuMetricEnum.GPU_FREE_MEM
                    or metric_type == GpuMetricEnum.GPU_USED_MEM
                    or metric_type == GpuMetricEnum.GPU_UTIL
                    or metric_type == GpuMetricEnum.GPU_TEMP
                ):
                    data = int(
                        rank_metric["dataMapByTime"][start_time]["count"]
                    )
                else:
                    data = round(
                        float(
                            rank_metric["dataMapByTime"][start_time]["count"]
                        ),
                        2,
                    )

                if pod not in job_metrics.keys():
                    job_metrics[pod] = GpuNodeMetric()
                if local_rank not in job_metrics[pod].node_metrics:
                    job_metrics[pod].node_metrics[local_rank] = GpuMetric()
                job_metrics[pod].node_metrics[local_rank].set_metric(
                    metric_type, data
                )

            return job_metrics

        except KeyError as e:
            logger.warning(f"Key error: {e}")
            traceback.print_exc()
            return None
        except ValueError as e:
            logger.warning(f"Value error: {e}")
            traceback.print_exc()
            return None
        except Exception as e:
            logger.warning(f"Unexpected error: {e}")
            traceback.print_exc()
            return None


class NpuMetricMonitor(SimpleMetricMonitor):
    """
    metric monitor of ascend NPU metrics

    """

    def __init__(self):
        super().__init__()

    def collect_job_metrics(
        self, job_name, metric_enum, start, end, pod_name=None, metrics=None
    ):
        rsp = self.query_job_metrics(
            job_name, metric_enum, start, end, is_gpu=False, pod_name=pod_name
        )
        if rsp is None:
            return None

        try:
            new_start, _ = self.adjust_timestamp(start, end)
            start_time = f"{int(new_start)}"
            job_metrics = {} if metrics is None else metrics

            for rank_metric in rsp["data"][0]:
                pod = rank_metric["tags"]["pod_name"]
                local_rank = int(rank_metric["tags"]["id"])
                if (
                    metric_enum == NpuMetricEnum.NPU_TOTAL_MEM
                    or metric_enum == NpuMetricEnum.NPU_USED_MEM
                    or metric_enum == NpuMetricEnum.NPU_UTIL
                    or metric_enum == NpuMetricEnum.NPU_TEMP
                    or metric_enum == NpuMetricEnum.NPU_HEALTH_STATE
                    or metric_enum == NpuMetricEnum.NPU_LINK_STATE
                    or metric_enum == NpuMetricEnum.NPU_OPTICAL_STATE
                    or metric_enum == NpuMetricEnum.NPU_NETWORK_STATE
                ):
                    data = int(
                        rank_metric["dataMapByTime"][start_time]["count"]
                    )
                else:
                    data = round(
                        float(
                            rank_metric["dataMapByTime"][start_time]["count"]
                        ),
                        2,
                    )

                if pod not in job_metrics.keys():
                    job_metrics[pod] = NpuNodeMetric()
                if local_rank not in job_metrics[pod].node_metrics:
                    job_metrics[pod].node_metrics[local_rank] = NpuMetric()
                job_metrics[pod].node_metrics[local_rank].set_metric(
                    metric_enum, data
                )

            return job_metrics

        except KeyError as e:
            logger.warning(f"Key error: {e}")
            traceback.print_exc()
            return None
        except ValueError as e:
            logger.warning(f"Value error: {e}")
            traceback.print_exc()
            return None
        except Exception as e:
            logger.warning(f"Unexpected error: {e}")
            traceback.print_exc()
            return None
