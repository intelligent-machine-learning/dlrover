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


class BaseModel(ABC):
    support_ts = False
    support_grad = False
    support_multi_output = False
    support_warm_start = False

    def __init__(self, num_cont: int, num_enum: int, num_out: int, **conf) -> None:
        """
        Base class for probabilistic regression models
        conf: configuration dict
        """
        self.num_cont = num_cont
        self.num_enum = num_enum
        self.num_out = num_out
        self.conf = conf
        assert self.num_cont >= 0
        assert self.num_enum >= 0
        assert self.num_out > 0
        assert self.num_cont + self.num_enum > 0
        if self.num_enum > 0:
            assert "num_uniqs" in self.conf
            assert isinstance(self.conf["num_uniqs"], list)
            assert len(self.conf["num_uniqs"]) == self.num_enum
        if not self.support_multi_output:
            assert self.num_out == 1, "Model only support single-output"

    @abstractmethod
    def fit(self, Xc: np.ndarray, Xe: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, Xc, Xe):
        """
        Return (possibly approximated) Gaussian predictive distribution
        Return py and ps2 where py is the mean and ps2 predictive variance.
        """
        pass

    @property
    def noise(self) -> np.ndarray:
        """
        Return estimated noise variance, for example, GP can view noise level
        as a hyperparameter and optimize it via MLE, another strategy could be
        using the MSE of training data as noise estimation

        Should return a (self.n_out, ) float tensor
        """
        return np.zeros(self.num_out)

    def sample_f(self):
        # Thompson sampling
        raise NotImplementedError("Thompson sampling is not supported")

    def sample_y(self, Xc, Xe, n_samples=1):
        py, ps2 = self.predict(Xc, Xe)
        ps = np.sqrt(ps2)
        samp = np.zeros((n_samples, py.shape[0], self.num_out))
        for i in range(n_samples):
            samp[i] = py + ps * np.random.randn(*py.shape)
        return samp
