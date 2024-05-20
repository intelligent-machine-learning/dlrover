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

import unittest
from typing import List

import torch
from botorch.test_functions import Hartmann

from dlrover.python.brain.hpsearch.base import RunResult
from dlrover.python.brain.hpsearch.bo import BayesianOptimizer

hartmann = Hartmann()
variance = 0.01


def obj(X):
    return -hartmann(X)


def observe(candidates, variance=0.0):
    for cand in candidates:
        cand.reward = (
            obj(torch.tensor(cand.parameters)) + torch.randn(1) * variance
        )
        cand.variance = variance


class BayesianOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = [[0.0, 1.0] for _ in range(hartmann.dim)]
        self.history: List[List[RunResult]] = []
        self.num_candidates = 3
        self.num_runs = 10

    def test_bo_without_variance(self):
        self.setUp()
        bo = BayesianOptimizer(
            self.bounds, self.history, self.num_candidates, use_variance=False
        )
        # Get random cold start candidates when history is empty
        cold_start_cands = bo.optimize()
        self.assertEqual(len(cold_start_cands), self.num_candidates)
        self.assertEqual(len(cold_start_cands[0].parameters), hartmann.dim)

        observe(cold_start_cands)
        self.history.append(cold_start_cands)

        best_observations = [max([cand.reward for cand in cold_start_cands])]

        # Perform BO loops here
        for _ in range(self.num_runs):
            cands = BayesianOptimizer(
                self.bounds,
                self.history,
                self.num_candidates,
                use_variance=False,
            ).optimize()
            observe(cands)
            best_observations.append(max([cand.reward for cand in cands]))
            self.history.append(cands)

        # This test could fail in a very very rare circumstances
        # In most cases, the best observation at the end should be the best
        self.assertGreater(best_observations[-1], best_observations[0])

    def test_bo_with_variance(self):
        self.setUp()
        bo = BayesianOptimizer(
            self.bounds, self.history, self.num_candidates, use_variance=True
        )
        # Get random cold start candidates when history is empty
        cold_start_cands = bo.optimize()
        self.assertEqual(len(cold_start_cands), self.num_candidates)
        self.assertEqual(len(cold_start_cands[0].parameters), hartmann.dim)

        observe(cold_start_cands)
        self.history.append(cold_start_cands)

        best_observations = [max([cand.reward for cand in cold_start_cands])]

        # Perform BO loops here
        for _ in range(self.num_runs):
            cands = BayesianOptimizer(
                self.bounds,
                self.history,
                self.num_candidates,
                use_variance=True,
            ).optimize()
            observe(cands)
            best_observations.append(max([cand.reward for cand in cands]))
            self.history.append(cands)

        # This test could fail in a very very rare circumstances
        # In most cases, the best observation at the end should be the best
        self.assertGreater(best_observations[-1], best_observations[0])
