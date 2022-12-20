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
from abc import ABCMeta, abstractmethod
from typing import List

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.singleton import singleton
from dlrover.python.master.stats.training_metrics import (
    DatasetMetric,
    ModelMetric,
    RuntimeMetric,
    TrainingHyperParams,
)


class JobMeta(object):
    def __init__(self, uuid, name="", namespace="", cluster="", user=""):
        self.uuid = uuid
        self.name = name
        self.namespace = namespace
        self.cluster = cluster
        self.user = user


class ReporterType(object):
    LOCAL = "local"
    DLROVER_BRAIN = "brain"


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
    def new_stats_reporter(cls, job_meta, reporter_type=None):
        if not reporter_type or reporter_type == ReporterType.LOCAL:
            return LocalStatsReporter(job_meta)
        else:
            logger.warning("Not support stats collector %s", reporter_type)


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

    def report_job_type(self, job_type: str):
        self._job_type = job_type

    def report_job_exit_reason(self, reason: str):
        self._exit_reason = reason

    def report_customized_data(self, data):
        self._custom_data = data

    def get_runtime_stats(self) -> List[RuntimeMetric]:
        return self._runtime_stats
