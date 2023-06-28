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


import numpy as np
import pandas as pd
from pymoo.config import Config
from pymoo.core.problem import Problem
from pymoo.factory import get_algorithm, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableCrossover, MixedVariableMutation
from pymoo.optimize import minimize

from atorch.auto.engine.sg_algo.hebo.acquisitions.acq import Acquisition
from atorch.auto.engine.sg_algo.hebo.design_space.design_space import DesignSpace

Config.show_compile_hint = False


class BOProblem(Problem):
    def __init__(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
        acq: Acquisition,
        space: DesignSpace,
        fix: dict = None,
    ) -> None:

        self.acq = acq
        self.space = space
        self.fix = fix  # NOTE: use self.fix to enable contextual BO

        super().__init__(len(lb), xl=lb, xu=ub, n_obj=acq.num_obj, n_constr=acq.num_constr)

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        num_x = x.shape[0]
        xcont = x[:, : self.space.num_numeric].astype(float)
        xenum = x[:, self.space.num_numeric :].round().astype(int)  # noqa: E203
        if self.fix is not None:
            df_x = self.space.inverse_transform(xcont, xenum)
            for k, v in self.fix.items():
                df_x[k] = v
            xcont, xenum = self.space.transform(df_x)
        tot_dims = self.acq.num_obj + self.acq.num_constr
        acq_eval = self.acq(xcont, xenum).reshape(num_x, tot_dims)
        out["F"] = acq_eval[:, : self.acq.num_obj]

        if self.acq.num_constr > 0:
            out["G"] = acq_eval[:, -1 * self.acq.num_constr :]  # noqa: E203


class EvolutionOpt:
    def __init__(self, design_space, acq, es=None, **conf):
        self.space = design_space
        self.es = es
        self.acq = acq
        self.pop = conf.get("pop", 128)
        self.iter = conf.get("iters", 100)
        self.verbose = conf.get("verbose", False)
        self.repair = conf.get("repair", None)
        self.sobol_init = conf.get("sobol_init", True)
        assert self.acq.num_obj > 0
        if self.es is None:
            self.es = "nsga2" if self.acq.num_obj > 1 else "ga"

    def get_init_pop(self, initial_suggest: pd.DataFrame = None) -> np.ndarray:
        if not self.sobol_init:
            init_pop = self.space.sample(self.pop)
        else:
            try:
                from scipy.stats.qmc import Sobol as SobolEngine

                eng = SobolEngine(self.space.num_paras, scramble=True)
                sobol_samp = eng.random(self.pop)
                opt_range = self.space.opt_ub - self.space.opt_lb
                sobol_samp = sobol_samp * opt_range + self.space.opt_lb
                x = sobol_samp[:, : self.space.num_numeric]
                xe = sobol_samp[:, self.space.num_numeric :].round().astype(int)  # noqa: E203
                for i, n in enumerate(self.space.numeric_names):
                    if self.space.paras[n].is_discrete_after_transform:
                        x[:, i] = x[:, i].round()
                init_pop = self.space.inverse_transform(x, xe)
            except ModuleNotFoundError:
                init_pop = self.space.sample(self.pop)

        if initial_suggest is not None:
            concat_pop = pd.concat([initial_suggest, init_pop], axis=0)
            init_pop = concat_pop.head(self.pop)
        x, xe = self.space.transform(init_pop)
        return np.hstack([x, xe.astype(float)])

    def get_crossover(self):
        mask = []
        for name in self.space.numeric_names + self.space.enum_names:
            if self.space.paras[name].is_discrete_after_transform:
                mask.append("int")
            else:
                mask.append("real")

        crossover = MixedVariableCrossover(
            mask,
            {
                "real": get_crossover("real_sbx", eta=15, prob=0.9),
                "int": get_crossover("int_sbx", eta=15, prob=0.9),
            },
        )
        return crossover

    def get_mutation(self):
        mask = []
        for name in self.space.numeric_names + self.space.enum_names:
            if self.space.paras[name].is_discrete_after_transform:
                mask.append("int")
            else:
                mask.append("real")

        mutation = MixedVariableMutation(
            mask,
            {
                "real": get_mutation("real_pm", eta=20),
                "int": get_mutation("int_pm", eta=20),
            },
        )
        return mutation

    def optimize(
        self,
        initial_suggest: pd.DataFrame = None,
        fix_input: dict = None,
        return_pop=False,
    ) -> pd.DataFrame:
        lb = self.space.opt_lb
        ub = self.space.opt_ub
        prob = BOProblem(lb, ub, self.acq, self.space, fix_input)
        init_pop = self.get_init_pop(initial_suggest)
        mutation = self.get_mutation()
        crossover = self.get_crossover()

        algo = get_algorithm(
            self.es,
            pop_size=self.pop,
            sampling=init_pop,
            mutation=mutation,
            crossover=crossover,
            repair=self.repair,
        )
        res = minimize(prob, algo, ("n_gen", self.iter), verbose=self.verbose)

        if res.X is not None and not return_pop:
            opt_x = res.X.reshape(-1, len(lb)).astype(float)
        else:
            opt_x = np.array([p.X for p in res.pop]).astype(float)
            if self.acq.num_obj == 1 and not return_pop:
                opt_x = opt_x[[np.random.choice(opt_x.shape[0])]]

        self.res = res
        opt_xcont = opt_x[:, : self.space.num_numeric]
        opt_xenum = opt_x[:, self.space.num_numeric :]  # noqa: E203
        df_opt = self.space.inverse_transform(opt_xcont, opt_xenum)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_opt[k] = v
        return df_opt
