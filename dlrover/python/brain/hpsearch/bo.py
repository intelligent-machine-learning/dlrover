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


import torch
from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from dlrover.python.brain.hpsearch.base import OptimizerBase, RunResult


class BayesianOptimizer(OptimizerBase):
    NUM_RESTARTS = 5
    RAW_SAMPLES = 256
    MC_SAMPLES = 256

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        history: list[list[RunResult]],
        num_candidates: int,
        use_variance: bool = False,
        **kwargs
    ) -> None:
        super().__init__(bounds, history, num_candidates, **kwargs)
        self.use_variance = use_variance

    def optimize(self):
        if self.cold_start:
            candidate = torch.empty([self.num_candidates, len(self.bounds)])
            for i, (lower_bound, upper_bound) in enumerate(self.bounds):
                torch.nn.init.uniform_(
                    candidate[:, i], a=lower_bound, b=upper_bound
                )
            candidate_ls = candidate.tolist()
        else:
            bounds = torch.tensor(self.bounds).T
            flattened = sum(self.history, [])
            train_x = torch.tensor(
                [run.parameters for run in flattened], dtype=torch.float64
            )  # n * d
            train_y = torch.tensor(
                [run.reward for run in flattened], dtype=torch.float64
            ).unsqueeze(
                1
            )  # n * 1

            input_transform = Normalize(d=len(flattened[0].parameters))
            outcome_transform = Standardize(m=1)

            if self.use_variance:
                train_yvar = torch.tensor(
                    [run.variance for run in flattened], dtype=torch.float64
                ).unsqueeze(
                    1
                )  # n * 1
                model = SingleTaskGP(
                    train_x,
                    train_y,
                    train_yvar,
                    outcome_transform=outcome_transform,
                    input_transform=input_transform,
                )
            else:
                model = SingleTaskGP(
                    train_x,
                    train_y,
                    outcome_transform=outcome_transform,
                    input_transform=input_transform,
                )

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            if self.use_variance:
                sampler = SobolQMCNormalSampler(torch.Size([self.MC_SAMPLES]))
                expected_improvement = qNoisyExpectedImprovement(
                    model, train_x, sampler
                )
            else:
                expected_improvement = qExpectedImprovement(
                    model, best_f=train_y.max()
                )

            num_restarts = self.NUM_RESTARTS
            raw_samples = self.RAW_SAMPLES

            candidate, acq_value = optimize_acqf(
                acq_function=expected_improvement,
                bounds=bounds,
                q=self.num_candidates,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        candidate_ls = candidate.tolist()
        runs = list(
            [RunResult(parameters=tuple(sample)) for sample in candidate_ls]
        )
        return runs
