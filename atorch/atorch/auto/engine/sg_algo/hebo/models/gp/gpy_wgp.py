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

import logging
import warnings

import GPy
import numpy as np
from GPy.models import InputWarpedGP
from GPy.util.input_warping_functions import KumarWarping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from atorch.auto.engine.sg_algo.hebo.models.base_model import BaseModel
from atorch.auto.engine.sg_algo.hebo.models.layers import OneHotTransform
from atorch.auto.engine.sg_algo.hebo.models.util import filter_nan

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GPyGP(BaseModel):
    """
    Input warped GP model implemented using GPy instead of GPyTorch

    Why doing so:
    - Input warped GP
    """

    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        total_dim = num_cont
        if num_enum > 0:
            self.one_hot = OneHotTransform(self.conf["num_uniqs"])
            total_dim += self.one_hot.num_out
        self.xscaler = MinMaxScaler((-1, 1))
        self.yscaler = StandardScaler()
        self.verbose = self.conf.get("verbose", False)
        self.num_epochs = self.conf.get("num_epochs", 200)
        self.warp = self.conf.get("warp", True)
        self.space = self.conf.get("space")  # DesignSpace
        self.num_restarts = self.conf.get("num_restarts", 20)
        if self.space is None and self.warp:
            warnings.warn("Space not provided, set warp to False")
            self.warp = False

        if self.warp:
            for i in range(total_dim):
                logging.getLogger(f"a{i}").disabled = True
                logging.getLogger(f"b{i}").disabled = True

    def fit_scaler(self, Xc: np.ndarray, y: np.ndarray):
        if Xc is not None and Xc.shape[1] > 0:
            if self.space is not None:
                opt_lb = self.space.opt_lb
                opt_ub = self.space.opt_ub
                num = self.space.num_numeric
                cont_lb = opt_lb[:num].astype("float").reshape([1, -1])
                cont_ub = opt_ub[:num].astype("float").reshape([1, -1])
                concat_x = np.concatenate([Xc, cont_lb, cont_ub], axis=0)
                self.xscaler.fit(concat_x)
            else:
                self.xscaler.fit(Xc)
        self.yscaler.fit(y)

    def trans(self, Xc: np.ndarray, Xe: np.ndarray, y: np.ndarray = None):
        if Xc is not None and Xc.shape[1] > 0:
            Xc_t = self.xscaler.transform(Xc)
        else:
            Xc_t = np.zeros((Xe.shape[0], 0))

        if Xe is None or Xe.shape[1] == 0:
            Xe_t = np.zeros((Xc.shape[0], 0))
        else:
            Xe_t = self.one_hot(Xe.astype("int"))
        Xall = np.concatenate([Xc_t, Xe_t], axis=1)

        if y is not None:
            y_t = self.yscaler.transform(y)
            return Xall, y_t
        return Xall

    def fit(self, Xc: np.ndarray, Xe: np.ndarray, y: np.ndarray):
        Xc, Xe, y = filter_nan(Xc, Xe, y, "all")
        self.fit_scaler(Xc, y)
        X, y = self.trans(Xc, Xe, y)

        k1 = GPy.kern.Linear(X.shape[1], ARD=False)
        k2 = GPy.kern.Matern32(X.shape[1], ARD=True)
        k2.lengthscale = np.std(X, axis=0).clip(min=0.02)
        k2.variance.set_prior(GPy.priors.Gamma(0.5, 1), warning=False)
        kern = k1 + k2
        if not self.warp:
            self.gp = GPy.models.GPRegression(X, y, kern)
        else:
            xmin = np.zeros(X.shape[1])
            xmax = np.ones(X.shape[1])
            xmin[: Xc.shape[1]] = -1
            warp_f = KumarWarping(X, Xmin=xmin, Xmax=xmax)
            self.gp = InputWarpedGP(X, y, kern, warping_function=warp_f)
        log_gauss_prior = GPy.priors.LogGaussian(-4.63, 0.5)
        self.gp.likelihood.variance.set_prior(log_gauss_prior, warning=False)

        self.gp.optimize_restarts(
            max_iters=self.num_epochs,
            verbose=self.verbose,
            num_restarts=self.num_restarts,
            robust=True,
        )
        return self

    def predict(self, Xc, Xe):
        Xall = self.trans(Xc, Xe)
        py, ps2 = self.gp.predict(Xall)
        mu = self.yscaler.inverse_transform(py.reshape((-1, 1)))
        var = self.yscaler.scale_**2 * ps2.reshape((-1, 1))
        return mu, np.clip(var, a_min=np.finfo(var.dtype).eps, a_max=None)

    def sample_f(self):
        raise NotImplementedError("Thompson sampling is not supported for GP")

    @property
    def noise(self):
        var_normalized = self.gp.likelihood.variance[0]
        return (var_normalized * self.yscaler.scale_**2).reshape(self.num_out)
