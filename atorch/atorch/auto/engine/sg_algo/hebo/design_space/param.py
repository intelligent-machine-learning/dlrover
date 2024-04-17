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
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
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
