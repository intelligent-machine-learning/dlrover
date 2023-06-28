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


from copy import deepcopy

import numpy as np
import pandas as pd
from numpy.random import choice
from sklearn.preprocessing import power_transform

from atorch.auto.engine.sg_algo.hebo.acq_optimizers import evolution_optimizer
from atorch.auto.engine.sg_algo.hebo.acquisitions.acq import MACE, Mean, Sigma
from atorch.auto.engine.sg_algo.hebo.models.model_factory import get_model

from .abstract_optimizer import AbstractOptimizer


class HEBO(AbstractOptimizer):
    support_parallel_opt = True
    support_combinatorial = True
    support_contextual = True

    def __init__(
        self,
        space,
        model_name="gpy",
        rand_sample=None,
        acq_cls=MACE,
        es="nsga2",
        model_config=None,
    ):
        """
        model_name  : surrogate model to be used
        rand_sample : iterations to perform random sampling
        """
        super().__init__(space)
        self.space = space
        self.es = es
        self.X = pd.DataFrame(columns=self.space.para_names)
        self.y = np.zeros((0, 1))
        self.model_name = model_name
        if rand_sample:
            self.rand_sample = max(2, rand_sample)
        else:
            self.rand_sample = 1 + self.space.num_paras

        self.sobol = None
        self.rand = None
        try:
            from scipy.stats.qmc import Sobol as SobolEngine

            self.sobol = SobolEngine(self.space.num_paras, scramble=False)
        except ModuleNotFoundError:
            from numpy.random import rand

            self.rand = rand

        self.acq_cls = acq_cls
        self._model_config = model_config

    def quasi_sample(self, n, fix_input=None):
        if self.sobol:
            samp = self.sobol.random(n)
        else:
            samp = self.rand(n, self.space.num_paras)
        opt_range = self.space.opt_ub - self.space.opt_lb
        samp = samp * opt_range + self.space.opt_lb
        x = samp[:, : self.space.num_numeric]
        xe = samp[:, self.space.num_numeric :]  # noqa: E203
        for i, n in enumerate(self.space.numeric_names):
            if self.space.paras[n].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        df_samp = self.space.inverse_transform(x, xe)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_samp[k] = v
        return df_samp

    @property
    def model_config(self):
        if self._model_config is None:

            if self.model_name == "gpy":
                cfg = {"verbose": False, "warp": True, "space": self.space}

            elif self.model_name == "rf":
                cfg = {"n_estimators": 20}
            else:
                cfg = {}
        else:
            cfg = deepcopy(self._model_config)

        if self.space.num_categorical > 0:
            paras = self.space.paras
            enum_names = self.space.enum_names
            cfg["num_uniqs"] = [paras[name].num_uniqs for name in enum_names]
        return cfg

    def get_best_id(self, fix_input: dict = None) -> int:
        if fix_input is None:
            return np.argmin(self.y.reshape(-1))
        X = self.X.copy()
        y = self.y.copy()
        for k, v in fix_input.items():
            if X[k].dtype != "float":
                crit = (X[k] != v).values
            else:
                crit = ((X[k] - v).abs() > np.finfo(float).eps).values
            y[crit] = np.inf
        if np.isfinite(y).any():
            return np.argmin(y.reshape(-1))
        else:
            return np.argmin(self.y.reshape(-1))

    def suggest(self, n_suggestions=1, fix_input=None):
        if self.acq_cls != MACE and n_suggestions != 1:
            raise RuntimeError("Only MACE can batch infer")
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            return sample
        else:
            X, Xe = self.space.transform(self.X)
            try:
                y_norm = self.y / self.y.std()
                if self.y.min() <= 0:
                    y = power_transform(y_norm, method="yeo-johnson")
                else:
                    y = power_transform(y_norm, method="box-cox")
                    if y.std() < 0.5:
                        y = power_transform(y_norm, method="yeo-johnson")

                if y.std() < 0.5:
                    raise RuntimeError("Power transformation failed")

            except RuntimeError:
                y = self.y.copy()

            num_numeric = self.space.num_numeric
            num_cat = self.space.num_categorical
            name = self.model_name
            cfg = self.model_config
            model = get_model(name, num_numeric, num_cat, 1, **cfg)
            model.fit(X, Xe, y)

            best_id = self.get_best_id(fix_input)
            best_x = self.X.iloc[[best_id]]
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best.squeeze()

            iter = max(1, self.X.shape[0] // n_suggestions)
            upsi = 0.5
            delta = 0.01
            bias = np.log(3 * np.pi**2 / (3 * delta))
            weight = 2.0 + self.X.shape[1] / 2.0
            ka = np.sqrt(upsi * 2 * (weight * np.log(iter) + bias))
            acq_config = {"best_y": py_best, "kappa": ka}
            acq = self.acq_cls(model, **acq_config)

            mu = Mean(model)
            sig = Sigma(model, linear_a=-1.0)
            opt = evolution_optimizer.EvolutionOpt(self.space, acq, es=self.es)

            opt_config = {"initial_suggest": best_x, "fix_input": fix_input}
            rec = opt.optimize(**opt_config).drop_duplicates()
            rec = rec[self.check_unique(rec)]

            cnt = 0
            while rec.shape[0] < n_suggestions:
                left_sug = n_suggestions - rec.shape[0]
                rand_rec = self.quasi_sample(left_sug, fix_input)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec = pd.concat([rec, rand_rec], ignore_index=True, axis=0)
                cnt += 1
                if cnt > 3:
                    break
            if rec.shape[0] < n_suggestions:
                left_sug = n_suggestions - rec.shape[0]
                rand_rec = self.quasi_sample(left_sug, fix_input)
                rec = pd.concat([rec, rand_rec], ignore_index=True, axis=0)

            rec_num = rec.shape[0]
            select_id = choice(rec_num, n_suggestions, False).tolist()

            py_all = mu(*self.space.transform(rec)).squeeze()
            ps_all = -1 * sig(*self.space.transform(rec)).squeeze()
            best_pred_id = np.argmin(py_all)
            best_unce_id = np.argmax(ps_all)
            if best_unce_id not in select_id and n_suggestions > 2:
                select_id[0] = best_unce_id
            if best_pred_id not in select_id and n_suggestions > 2:
                select_id[1] = best_pred_id
            rec_selected = rec.iloc[select_id].copy()
            return rec_selected

    def check_unique(self, rec):
        conat_df = pd.concat([self.X, rec], axis=0)
        return (~conat_df.duplicated().tail(rec.shape[0]).values).tolist()

    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : pandas DataFrame
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,1)
            Corresponding values where objective has been evaluated
        """
        valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX = X.iloc[valid_id]
        yy = y[valid_id].reshape(-1, 1)
        self.X = pd.concat([self.X, XX], ignore_index=True, axis=0)
        self.y = np.vstack([self.y, yy])

    @property
    def best_x(self) -> pd.DataFrame:
        if self.X.shape[0] == 0:
            raise RuntimeError("No data has been observed!")
        else:
            return self.X.iloc[[self.y.argmin()]]

    @property
    def best_y(self) -> float:
        if self.X.shape[0] == 0:
            raise RuntimeError("No data has been observed!")
        else:
            return self.y.min()
