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


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RunResult:
    parameters: Tuple = ()
    reward: float = 0.0
    variance: float = 0.01
    epsilon: float = 0.0


class OptimizerBase(ABC):
    """Base class for hyperparameter optimization

    Args:
        bounds (List[Tuple[float, float]]): A list of tuples representing the lower and
            upper bounds for each input dimension.
        history (List[List[RunResult]]): A list of lists containing the previously evaluated
            points and their corresponding objective values.
        num_candidates (int): The number of candidates to propose at each iteration.
    """  # noqa: E501

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        history: List[List[RunResult]],
        num_candidates: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.bounds = bounds
        self.history = history
        self.num_candidates = num_candidates
        self.cold_start = history is None or len(sum(self.history, [])) == 0

    @abstractmethod
    def optimize(self) -> List[RunResult]:
        pass
