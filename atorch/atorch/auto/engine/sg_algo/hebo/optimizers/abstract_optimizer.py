# Copyright 2022 The ElasticDL Authors. All rights reserved.
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

import numpy as np
import pandas as pd

from atorch.auto.engine.sg_algo.hebo.design_space.design_space import DesignSpace


class AbstractOptimizer(ABC):
    support_parallel_opt = False
    support_constraint = False
    support_multi_objective = False
    support_combinatorial = False
    support_contextual = False

    def __init__(self, space: DesignSpace) -> None:
        self.space = space

    @abstractmethod
    def suggest(self, n_suggestions=1, fix_input: dict = None):
        """
        Perform optimisation and give recommendation using data observed so far
        ---------------------
        n_suggestions:  number of recommendations in this iteration

        fix_input:      parameters NOT to be optimized, but rather fixed, this
                        can be used for contextual BO.
        """
        pass

    @abstractmethod
    def observe(self, x: pd.DataFrame, y: np.ndarray):
        """
        Observe new data
        """
        pass

    @property
    @abstractmethod
    def best_x(self) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def best_y(self) -> float:
        pass
