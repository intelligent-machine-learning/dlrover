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

from typing import List

from dlrover.python.diagnosis.common.inference_chain import (
    Inference,
    InferenceOperator,
)


class SimpleOperator(InferenceOperator):
    """
    SimpleOperator is a simple implementation
    of InferenceOperator
    """

    def __init__(self):
        super().__init__(None)

    def is_compatible(self, inference: Inference) -> bool:
        if inference.name == "simple_problem":
            return True
        else:
            return False

    def infer(self, inferences: List[Inference]) -> List[Inference]:
        return [
            Inference(
                name="simple_result",
            )
        ]
