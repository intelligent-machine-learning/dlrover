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

from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.diagnosis_action import (
    DiagnosisAction,
    NoAction,
)
from dlrover.python.diagnosis.common.inference_chain import Coordinator
from dlrover.python.diagnosis.inferencechain.coordinator import (
    coordinate_solutions,
)
from dlrover.python.diagnosis.inferencechain.inference_chain import (
    Inference,
    InferenceChain,
    InferenceOperator,
    combine_inferences,
)


class Diagnostician:
    """
    Diagnostician is to observe training problems and explore solutions to
    those problems during training.
    """

    def __init__(self, data_manager):
        self._data_manager = data_manager
        self._training_problems: List[Inference] = []
        self._observers: List[InferenceOperator] = []
        self._resolvers: List[InferenceOperator] = []

    def get_observing_operators(self) -> List[InferenceOperator]:
        return self._observers

    def get_resolving_operators(self) -> List[InferenceOperator]:
        return self._resolvers

    def register_precheck_(self, problems: List[Inference]):
        self._training_problems = problems

    def register_training_problems(self, problems: List[Inference]):
        self._training_problems = problems

    def register_observers(self, operators: List[InferenceOperator]):
        self._observers = operators

    def register_resolvers(self, operators: List[InferenceOperator]):
        self._resolvers = operators

    def observe_training(self) -> List[Inference]:
        """
        To check if any problem in _training_problems happen

        Return:
            observed problems
        """
        if len(self._training_problems) == 0:
            logger.debug("No training problem is registered.")
            return []
        combined_problems: List[Inference] = []
        for problem in self._training_problems:
            logger.debug(f"Observing problem: {problem}")
            ic = InferenceChain([problem], self._observers)
            observed_problems = ic.infer()
            combined_problems = combine_inferences(
                combined_problems, observed_problems
            )
        return combined_problems

    def resolve_problems(self, problems: List[Inference]) -> DiagnosisAction:
        """
        Generate the diagnosis action for observed problem

        Args:
            problems: observed(combined) problems

        Return:
            diagnosis action
        """
        combined_solutions: List[Inference] = []
        for problem in problems:
            logger.debug(f"Resolving problem: {problem}")
            ic = InferenceChain([problem], self._resolvers)
            input_solutions = ic.infer()
            if len(input_solutions) > 0:
                combined_solutions = combine_inferences(
                    combined_solutions, input_solutions
                )

        return coordinate_solutions(combined_solutions)

    def diagnosis(self, problems: List[Inference], operators: List[InferenceOperator],
                  coordinator: Coordinator) -> DiagnosisAction:
        ic = InferenceChain(problems, operators)
        conclusions = ic.infer()
        if len(conclusions) > 0:
            try:
                return coordinator.coordinate(conclusions)
            except Exception as e:
                logger.error(f"fail to generate action for {conclusions}: {e}")
                return NoAction()
        return NoAction()


