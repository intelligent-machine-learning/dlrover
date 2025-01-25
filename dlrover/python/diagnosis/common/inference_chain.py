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
from dataclasses import dataclass, field
from typing import Dict, List
from dlrover.python.diagnosis.common.diagnosis_action import DiagnosisAction


class InferenceName:
    NONE = "none"
    TRAINING = "training"
    NODE = "node"
    WORKER = "worker"
    ACTION = "action"
    RESOURCE = "resource"
    GPU = "gpu"


class InferenceAttribute:
    ISORNOT = "is_or_not"
    IS = "is"
    NOT = "not"
    COLLECT = "collect"


class InferenceDescription:
    NONE = "n/a"
    HANG = "hang"
    FAILURE = "failure"
    METRICS = "metrics"
    EVENT = "event"
    ERROR = "error"
    RESOURCE = "resource"


@dataclass
class Inference(object):
    """
    Inference object reflects problems and failures during training.
    """

    name: str = ""
    attribution: str = ""
    description: str = ""
    configs: Dict[str, str] = field(default_factory=dict)


class InferenceOperator(metaclass=ABCMeta):
    """
    InferenceOperator is used to infer the root cause of problems.
    """

    def __init__(self, data_manager):
        self._data_manager = data_manager

    @abstractmethod
    def infer(self, inferences: List[Inference]) -> List[Inference]:
        pass

    # check if the InferenceOperator can be used to infer a given inference
    @abstractmethod
    def is_compatible(self, inference: Inference) -> bool:
        pass

    @property
    def data_manager(self):
        return self._data_manager


class Coordinator(metaclass=ABCMeta):
    """
    Coordinator is to coordinate multiple inferences and generate the final action.
    """

    def __init__(self):
        pass

    @abstractmethod
    def coordinate(self, inferences: List[Inference]) -> DiagnosisAction:
        pass


def is_same_inference(inference1: Inference, inference2: Inference) -> bool:
    if (
        inference1.name == inference2.name
        and inference1.attribution == inference2.attribution
        and inference1.description == inference2.description
    ):
        return True
    else:
        return False


def is_inference_included(infs: List[Inference], inf: Inference) -> bool:
    if not infs or not inf:
        return False
    for i in infs:
        if is_same_inference(i, inf):
            return True
    return False


def combine_inferences(
    inferences1: List[Inference], inferences2: List[Inference]
) -> List[Inference]:
    inferences = []
    for inference2 in inferences2:
        if not is_inference_included(inferences1, inference2):
            inferences.append(inference2)

    for inference1 in inferences1:
        inferences.append(inference1)

    return inferences


def is_training_hanged(inf: Inference):
    if (
        inf.name == InferenceName.TRAINING
        and inf.attribution == InferenceAttribute.IS
        and inf.description == InferenceDescription.HANG
    ):
        return True
    return False
