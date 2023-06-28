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


class Parameter(ABC):
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.name = param_dict["name"]
        pass

    @abstractmethod
    def sample(self, num=1) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def inverse_transform(self, x: np.array) -> np.array:
        pass

    @property
    @abstractmethod
    def is_numeric(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """
        Integer and categorical variable
        """
        pass

    @property
    @abstractmethod
    def is_discrete_after_transform(self) -> bool:
        pass

    @property
    def is_categorical(self) -> bool:
        return not self.is_numeric

    @property
    @abstractmethod
    def opt_lb(self) -> float:
        pass

    @property
    @abstractmethod
    def opt_ub(self) -> float:
        pass
