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

from atorch.auto.engine.sg_algo.sg_algorithm import StrategyGenerationAlgorithm

""" combination strategy generation algorithm
Algorithm will generate new candidate strategies and add in executor.strateies
Input:
  executor: the executor contains information for current acceleration
            optimitzation status.
Output:
  is_done: bool indicating if the algorithm finishs after this call.

"""


class CombinationAlgorithm(StrategyGenerationAlgorithm):
    def __init__(self):
        super().__init__("combination_sg")
        self.new_strategy_num = 0

    def add_strategy(self, executor, strategy):
        if executor.strategy_infos.add_strategy(strategy):
            self.new_strategy_num += 1

    def strategy_generate(self, executor):
        # Return is_done, tasks, new_strategy_num
        if self.is_done:
            return True, None, 0
        # create a list of strategies by optimization method combination.
        tasks = None
        self.new_strategy_num = 0

        can_module_replace = not executor.opt_method_lib.methods["module_replace"].disabled
        module_replace_strategy = [("module_replace", pickle.dumps(None), False)]

        amp_native_strategy = [("amp_native", pickle.dumps(None), False)]
        amp_apex_o2_strategy = [("amp_apex_o2", pickle.dumps(None), False)]
        zero2_strategy = [("zero2", pickle.dumps(None), False)]
        fsdp_strategy = [("fsdp", pickle.dumps(None), False)]

        zeros = [[], zero2_strategy, fsdp_strategy]
        others = [[], amp_native_strategy]
        if can_module_replace:
            others.append(module_replace_strategy + amp_native_strategy)

        if executor.total_process > 1:
            p_mode = ([("data", executor.total_process)], None)
            dp_strategy = [("parallel_mode", pickle.dumps(p_mode), False)]
            self.add_strategy(executor, dp_strategy + amp_apex_o2_strategy)
            for zero_s in zeros:
                for other_s in others:
                    strategy = dp_strategy + zero_s + other_s
                    if len(strategy) > 1:
                        self.add_strategy(executor, strategy)
        else:
            self.add_strategy(executor, amp_apex_o2_strategy)
            for other_s in others:
                if len(other_s) > 0:
                    self.add_strategy(executor, other_s)

        self.is_done = True
        return True, tasks, self.new_strategy_num
