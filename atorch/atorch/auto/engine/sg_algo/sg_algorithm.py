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


class StrategyGenerationAlgorithm(object):
    """A strategy generation (SG) algorithm implementation.
    Call strategy_generate with executor to generate candidate strategies.
    strategy_generate can be called multiple times to generate strategies in
    multiple stages.
    """

    def __init__(self, name=None):
        self.name = name
        self.is_done = False

    def strategy_generate(self, _):
        """
        Input: executor which contains optimization method, strategies.
        The output is 3-tuple:
        is_done: bool incidating if the algorithm finishs after this call.
        tasks: None or list(task), new tasks to execute.
        new_strategy_num: int for the number of new strategy added.
        """
        self.is_done = True
        return self.is_done, None, 0

    def __call__(self, executor):
        return self.strategy_generate(executor)
