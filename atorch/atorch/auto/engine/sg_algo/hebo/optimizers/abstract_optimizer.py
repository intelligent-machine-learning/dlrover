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
