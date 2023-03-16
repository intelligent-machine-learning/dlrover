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

import copy
import json
from abc import ABCMeta, abstractmethod
from typing import List

from dlrover.proto import brain_pb2
from dlrover.python.brain.client import GlobalBrainClient
from dlrover.python.common.constants import ReporterType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import singleton
from dlrover.python.master.stats.training_metrics import (
    DatasetMetric,
    ModelMetric,
    RuntimeMetric,
    TrainingHyperParams,
)

DATA_STORE = "base_datastore"
_MAX_SAMPLE_NUM_PER_STAT = 5


class JobMeta(object):
    def __init__(self, uuid, name="", namespace="", cluster="", user=""):
        self.uuid = uuid
        self.name = name
        self.namespace = namespace
        self.cluster = cluster
        self.user = user


def init_job_metrics_message(job_meta: JobMeta):
    job_metrics = brain_pb2.JobMetrics()
    job_metrics.data_store = DATA_STORE
    job_metrics.job_meta.uuid = job_meta.uuid
    job_metrics.job_meta.name = job_meta.name
    job_metrics.job_meta.user = job_meta.user
    job_metrics.job_meta.cluster = job_meta.cluster
    job_metrics.job_meta.namespace = job_meta.namespace
    return job_metrics


class StatsReporter(metaclass=ABCMeta):
    def __init__(self, job_meta: JobMeta):
        self._job_meta = job_meta

    @abstractmethod
    def report_dataset_metric(self, metric: DatasetMetric):
        pass

    @abstractmethod
    def report_training_hyper_params(self, params: TrainingHyperParams):
        pass

    @abstractmethod
    def report_model_metrics(self, metric: ModelMetric):
        pass

    @abstractmethod
    def report_runtime_stats(self, stats: RuntimeMetric):
        pass

    @abstractmethod
    def report_job_type(self, job_type: str):
        pass

    @abstractmethod
    def report_job_exit_reason(self, reason: str):
        pass

    @abstractmethod
    def report_customized_data(self, data):
        pass

    @classmethod
    def new_stats_reporter(cls, job_meta, reporter=None):
        if not reporter or reporter == ReporterType.LOCAL:
            logger.info("New local stats reporter.")
            return LocalStatsReporter(job_meta)
        elif reporter == ReporterType.DLROVER_BRAIN:
            logger.info("New brain stats reporter.")
            return BrainReporter(job_meta)
        else:
            logger.warning("Not support stats collector %s.", reporter)


@singleton
class LocalStatsReporter(StatsReporter):
    def __init__(self, job_meta):
        self._job_meta = job_meta
        self._runtime_stats: List[RuntimeMetric] = []
        self._dataset_metric = None
        self._training_hype_params = None
        self._model_metric = None
        self._job_type = ""
        self._exit_reason = ""
        self._custom_data = None

    def report_dataset_metric(self, metric: DatasetMetric):
        self._dataset_metric = metric

    def report_training_hyper_params(self, params: TrainingHyperParams):
        self._training_hype_params = params

    def report_model_metrics(self, metric: ModelMetric):
        self._model_metric = metric

    def report_runtime_stats(self, stats: RuntimeMetric):
        self._runtime_stats.append(copy.deepcopy(stats))
        latest_stat = self._runtime_stats[-1]
        index = 0
        for i, stat in enumerate(self._runtime_stats):
            if len(stat.running_nodes) == len(latest_stat.running_nodes):
                index = i
                break
        if len(self._runtime_stats) - index > _MAX_SAMPLE_NUM_PER_STAT:
            new_stats = self._runtime_stats[0:index]
            start_index = len(self._runtime_stats) - _MAX_SAMPLE_NUM_PER_STAT
            new_stats.extend(self._runtime_stats[start_index:])
            self._runtime_stats = new_stats

    def report_job_type(self, job_type: str):
        self._job_type = job_type

    def report_job_exit_reason(self, reason: str):
        self._exit_reason = reason

    def report_customized_data(self, data):
        self._custom_data = data

    def get_runtime_stats(self) -> List[RuntimeMetric]:
        return self._runtime_stats


@singleton
class BrainReporter(StatsReporter):
    def __init__(self, job_meta: JobMeta) -> None:
        self._job_meta = job_meta
        self._brain_client = GlobalBrainClient.BRAIN_CLIENT

    def report_dataset_metric(self, dataset: DatasetMetric):
        self._brain_client.report_training_set_metric(self._job_meta, dataset)

    def report_training_hyper_params(self, params: TrainingHyperParams):
        self._brain_client.report_training_hyper_params(self._job_meta, params)

    def report_model_metrics(self, metric: ModelMetric):
        """Report the model meta to EasyDL DB.
        Args:
            job_uuid: str, the unique id of the job which is usually
                the uuid in the job yaml of k8s.
            model size: int, the size of the NN model.
            variable_count: int, the total count of variables in the model.
            ops_count: int, the total count of ops in the model.
        """
        job_metrics = init_job_metrics_message(self._job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Model_Feature
        metrics = job_metrics.model_feature
        metrics.total_variable_size = metric.tensor_stats.total_variable_size
        metrics.variable_count = metric.tensor_stats.variable_count
        metrics.op_count = metric.op_stats.op_count
        self._brain_client.report_metrics(job_metrics)

    def report_runtime_stats(self, stats: RuntimeMetric):
        self._brain_client.report_node_runtime_stats(
            self._job_meta, self._job_meta.namespace, stats
        )

    def report_job_type(self, job_type: str):
        """Report the job type to EasyDL DB.
        Args:
            job_uuid: str, the unique id of the job which is usually
                the uuid in the job yaml of k8s.
            job_type: str, the type of training job like "alps", "atorch",
                "penrose" and so on.
        """
        job_metrics = init_job_metrics_message(self._job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Type
        job_metrics.type = job_type
        logger.info("Report job_type = %s", job_type)
        self._brain_client.report_metrics(job_metrics)

    def report_job_exit_reason(self, reason: str):
        self._brain_client.report_job_exit_reason(self._job_meta, reason)

    def report_customized_data(self, data):
        """Report the job resource to EasyDL DB.
        Args:
            job_uuid: str, the unique id of the job which is usually
                the uuid in the job yaml of k8s.
            cutomized_data: A dictionary.
        """
        job_metrics = init_job_metrics_message(self._job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Customized_Data
        job_metrics.customized_data = json.dumps(data)
        self._brain_client.report_metrics(job_metrics)

    def report_job_meta(self):
        """Report the job meta to EasyDL DB.
        Args:
            job_uuid: str, the unique id of the job which is usually
                the uuid in the job yaml of k8s.
            job_name: str, the name of training job.
            user_id: the user id.
        """
        job_metrics = init_job_metrics_message(self._job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Workflow_Feature
        metrics = job_metrics.workflow_feature
        metrics.job_name = self._job_meta.name
        metrics.user_id = self._job_meta.user
        self._brain_client.report_metrics(job_metrics)

    def report_job_resource(self, job_resource):
        """Report the job resource to EasyDL DB.
        Args:
            job_uuid: str, the unique id of the job which is usually
                the uuid in the job yaml of k8s.
            job_resource: dlrover.python.master.resource.JobResource instance.
        """
        job_metrics = init_job_metrics_message(self._job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Resource
        job_metrics.resource = job_resource.toJSON()
        self._brain_client.report_metrics(job_metrics)
