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

from diagnosis.inferencechain.inferenceoperator.resolver.resolve_training_hang_operator import (  # noqa: E501
    ResolveTrainingHangOperator,
)

from dlrover.python.diagnosis.inferencechain.inference_chain import (
    Inference,
    InferenceChain,
    InferenceOperator,
)
from dlrover.python.diagnosis.inferencechain.inferenceoperator.observer.check_training_hang_operator import (  # noqa: E501
    CheckTrainingHangOperator,
)


class Diagnostician:
    def __init__(self, data_manager):
        self._data_manager = data_manager
        self._pre_checks: List[Inference] = []
        self._training_problems: List[Inference] = []

    def get_pre_check_operators(self) -> List[InferenceOperator]:
        return []

    def get_observing_operators(self) -> List[InferenceOperator]:
        return [CheckTrainingHangOperator(self._data_manager)]

    def get_resolving_operators(self) -> List[InferenceOperator]:
        return [ResolveTrainingHangOperator(self._data_manager)]

    def register_pre_check(self, pre_checks: List[Inference]):
        self._pre_checks = pre_checks

    def register_problems(self, problems: List[Inference]):
        self._training_problems = problems

    def check_training(self) -> List[Inference]:
        ic = InferenceChain(self._pre_checks, self.get_pre_check_operators())
        return ic.infer()

    def observe_training(self) -> List[Inference]:
        ic = InferenceChain(
            self._training_problems, self.get_observing_operators()
        )
        return ic.infer()

    def diagnose_failure(self, observed_result: Inference) -> List[Inference]:
        ic = InferenceChain([observed_result], self.get_resolving_operators())
        return ic.infer()
