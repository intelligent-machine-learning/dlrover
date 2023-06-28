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
import pickle
import warnings

import numpy as np
import pandas as pd

from atorch.auto.engine.sg_algo.hebo.design_space.design_space import DesignSpace
from atorch.auto.engine.sg_algo.hebo.optimizers.hebo import HEBO
from atorch.auto.engine.sg_algo.sg_algorithm import StrategyGenerationAlgorithm
from atorch.auto.engine.sg_algo.utils import (
    analyse_strategies,
    gen_space_config,
    get_finished_strategies,
    rec_to_easydl_strategy,
    transform_finished_strategies_to_hebo,
    unfeasible_filter,
)
from atorch.common.log_utils import default_logger as logger

""" combination strategy generation algorithm
Algorithm will generate new candidate strategies and add in executor.strateies
Input:
  executor: the executor contains information for current acceleration
            optimitzation status.
Output:
  is_done: bool indicating if the algorithm finishs after this call.

"""

NOT_COMPATIBLE_OPTS = [
    "amp_apex_o1+zero2_fairscale",
    "amp_apex_o2+zero2_fairscale",
    "amp_apex_o1+zero2_fsdp",
    "amp_apex_o2+zero2_fsdp",
    "amp_apex_o1+fsdp",
    "amp_apex_o2+fsdp",
    "amp_apex_o2+zero1",
]


class BOAlgorithm(StrategyGenerationAlgorithm):
    def __init__(self, max_iter=30, max_patience=8):
        super().__init__("bo_sg")
        self.new_strategy_num = 0
        self.max_iter = max_iter
        self.max_patience = max_patience
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    def add_strategy(self, executor, strategy):
        conf = {"skip_duplicate": False, "is_baseline": False}
        if executor.strategy_infos.add_strategy(strategy, **conf):
            self.new_strategy_num += 1

    def _bo_search(self, executor, included_opts, finished_strategies):
        """
        Inputs:
            excutor: easydl excutor
            included_opts: included_opts from auto acc
            finished_strategies:  finished easydl strategies so far

        Outputs:
            rec: pd.DataFrame, bo recommendation
            best_y: float, highest throughput so far
        """
        groups = executor.opt_method_lib.groups
        methods = executor.opt_method_lib.methods

        config = gen_space_config(groups, methods, executor.total_process, included_opts)
        # get the bo search space
        space = DesignSpace().parse(config)
        # initilize bo optimizer
        opt = HEBO(space)

        # transform edl strategy to hebo format
        X, y = transform_finished_strategies_to_hebo(space, groups, finished_strategies, included_opts)

        cnt = 0
        # update the dataset of bo optimizer
        opt.observe(X, y)

        # verbose bo searching
        if executor.verbose:
            logger.info(X)
            logger.info(y)

        # if the space is small and a few forbidden strategies exist,
        # bo could get no feasible strategy
        # hence, we try 3 times
        while True:
            # bo recommends strategy
            rec = opt.suggest(n_suggestions=len(NOT_COMPATIBLE_OPTS) + 3)
            # filter the unfeasible combinations
            rec, rec_status = unfeasible_filter(rec, ["amp", "zero"], NOT_COMPATIBLE_OPTS)
            # if bo gets feasible strategy, break
            if rec_status:
                break
            else:
                opt.observe(rec, np.array([[0.0]]))

            cnt += 1

            if cnt > 3:
                # after 3 times, output baseline
                rec = pd.DataFrame(
                    columns=space.para_names,
                    data=["NotChosen"] * len(space.para_names),
                )
                break

        return rec, opt.best_y

    def strategy_generate(self, executor):
        # Return is_done, tasks, new_strategy_num
        if self.is_done:
            return True, None, 0
        # get wrap cls
        wrap_cls = executor.analyser_result.get("opt_config_submodule_names")

        tasks = None
        self.new_strategy_num = 0

        # get all the finished strategies
        strategies = executor.strategy_infos.strategies
        finished_strategies = get_finished_strategies(strategies)

        # get the iterations of bo search and patience of early stopping
        iterations, patience = analyse_strategies(finished_strategies)

        if not executor.planner.included_opts:
            included_opts = []
        else:
            included_opts = executor.planner.included_opts

        # 30 iterations at most
        # early stop patience threshold is 8
        if iterations > self.max_iter or patience > self.max_patience:
            self.is_done = True
        else:
            # get the bo search config

            rec, best_y = self._bo_search(
                executor,
                included_opts,
                finished_strategies,
            )
            edl_rec = rec_to_easydl_strategy(rec, wrap_cls, included_opts)

            # verbose
            if executor.verbose:
                debug_strategy = []
                for strategy in edl_rec[0]:
                    config = pickle.loads(strategy[1])
                    debug_strategy.append((strategy[0], config, strategy[2]))
                logger.info(debug_strategy)

            for strategy in edl_rec:
                self.add_strategy(executor, strategy)
            try:
                logger.info(f"bo iters:{iterations} with patiences:{patience}")
                logger.info(f"best throughput is {-1 * best_y}")
            except RuntimeError:
                logger.info(f"bo iters:{iterations} with patiences:{patience}")

        return False, tasks, self.new_strategy_num
