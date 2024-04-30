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
from dataclasses import dataclass
from typing import List


class InferenceName:
    END = "end"
    TRAINING = "training"


class InferenceAttribute:
    ISORNOT = "is_or_not"


class InferenceDescription:
    HANG = "hang"


@dataclass
class Inference(object):
    """
    Inference object reflects problems and failures during training.
    """

    name: str = ""
    attribution: str = ""
    description: str = ""


class InferenceOperator(metaclass=ABCMeta):
    """
    InferenceOperator is used to infer the root cause of problems.
    """

    def __init__(self):
        pass

    @abstractmethod
    def infer(self, inferences: List[Inference]) -> List[Inference]:
        pass

    # check if the operator can be used to infer a given inference
    @abstractmethod
    def is_compatible(self, inference: Inference) -> bool:
        pass


def same_inference(inference1: Inference, inference2: Inference) -> bool:
    if (
        inference1.name == inference2.name
        and inference1.attribution == inference2.attribution
        and inference1.description == inference2.description
    ):
        return True
    else:
        return False


def combine_inferences(
    inferences1: List[Inference], inferences2: List[Inference]
) -> List[Inference]:
    inferences = []
    for inference2 in inferences2:
        is_duplicate = False
        for inference1 in inferences1:
            if same_inference(inference1, inference2):
                is_duplicate = True
                break
        if not is_duplicate:
            inferences.append(inference2)

    for inference1 in inferences1:
        inferences.append(inference1)

    return inferences
