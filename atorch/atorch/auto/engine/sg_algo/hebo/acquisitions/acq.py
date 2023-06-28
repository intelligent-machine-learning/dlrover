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
from numpy.random import randn
from scipy.stats import norm

from atorch.auto.engine.sg_algo.hebo.models.base_model import BaseModel


class Acquisition(ABC):
    def __init__(self, model, **conf):
        self.model = model

    @property
    @abstractmethod
    def num_obj(self):
        pass

    @property
    @abstractmethod
    def num_constr(self):
        pass

    @abstractmethod
    def eval(self, x: np.ndarray, xe: np.ndarray) -> np.ndarray:
        """
        Shape of output tensor: (x.shape[0], self.num_obj + self.num_constr)
        """
        pass

    def __call__(self, x: np.ndarray, xe: np.ndarray):
        return self.eval(x, xe)


class SingleObjectiveAcq(Acquisition):
    """
    Single-objective, unconstrained acquisition
    """

    def __init__(self, model: BaseModel, **conf) -> None:
        super().__init__(model, **conf)

    @property
    def num_obj(self):
        return 1

    @property
    def num_constr(self):
        return 0


class Mean(SingleObjectiveAcq):
    def __init__(self, model: BaseModel, **conf) -> None:
        super().__init__(model, **conf)
        assert model.num_out == 1

    def eval(self, x: np.ndarray, xe: np.ndarray) -> np.ndarray:
        py, _ = self.model.predict(x, xe)
        return py


class Sigma(SingleObjectiveAcq):
    def __init__(self, model: BaseModel, **conf) -> None:
        super().__init__(model, **conf)
        assert model.num_out == 1

    def eval(self, x: np.ndarray, xe: np.ndarray) -> np.ndarray:
        _, ps2 = self.model.predict(x, xe)
        return -1 * np.sqrt(ps2)


class MACE(Acquisition):
    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get("kappa", 2.0)
        self.eps = conf.get("eps", 1e-4)
        self.tau = best_y

    @property
    def num_constr(self):
        return 0

    @property
    def num_obj(self):
        return 3

    def eval(self, x: np.ndarray, xe: np.ndarray) -> np.ndarray:
        """
        minimize (-1 * EI,  -1 * PI, lcb)
        """

        py, ps2 = self.model.predict(x, xe)
        noise = np.sqrt(2.0 * self.model.noise)
        ps = np.clip(np.sqrt(ps2), a_min=np.finfo(ps2.dtype).eps, a_max=None)
        lcb = (py + noise * randn(*py.shape)) - self.kappa * ps

        normed = (self.tau - self.eps - py - noise * randn(*py.shape)) / ps
        dist = norm()
        log_phi = dist.logpdf(normed)
        Phi = dist.cdf(normed)
        PI = Phi
        EI = ps * (Phi * normed + np.exp(log_phi))
        # https://stats.stackexchange.com/questions/106003/approximation-of-logarithm-of-standard-normal-cdf-for-x0
        logEIapp = np.log(ps) - 0.5 * normed**2 - np.log(normed**2 - 1)
        logpi_bias = np.log(np.sqrt(2 * np.pi))
        logPIapp = -0.5 * normed**2 - np.log(-1 * normed) - logpi_bias

        EI_valid = np.isfinite(np.log(EI))
        PI_valid = np.isfinite(np.log(PI))
        use_app = ~((normed > -6) & EI_valid & PI_valid).reshape(-1)
        out = np.zeros((x.shape[0], 3))
        out[:, 0] = lcb.reshape(-1)
        out[:, 1][use_app] = -1 * logEIapp[use_app].reshape(-1)
        out[:, 2][use_app] = -1 * logPIapp[use_app].reshape(-1)
        out[:, 1][~use_app] = -1 * np.log(EI[~use_app]).reshape(-1)
        out[:, 2][~use_app] = -1 * np.log(PI[~use_app]).reshape(-1)
        return out
