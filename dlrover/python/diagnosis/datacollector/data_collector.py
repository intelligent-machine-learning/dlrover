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


class CollectorType:
    CUDALOG = "cuda_log"
    TRAININGLOG = "training_log"
    CHIPMETRICS = "chip_metrics"


class DataCollector(metaclass=ABCMeta):
    """
    DataCollector collects certain type of data and report to master.
    Those data is used to diagnosis the faults of training.
    """

    def __init__(self):
        pass

    @abstractmethod
    def collect_data(self) -> object:
        pass

    @abstractmethod
    def to_collect_data(self) -> bool:
        pass
