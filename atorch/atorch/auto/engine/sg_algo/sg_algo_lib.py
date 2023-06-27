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


from atorch.auto.engine.sg_algo.combination_sg import CombinationAlgorithm


class StrategyGenerationAlgorithmLibrary(object):
    """
    Each strategy generation (SG) algorithm is a StrategyGenerationAlgorithm
    instance, which can be called to generate new strategies.
    """

    def __init__(self):
        self.algorithms = {}
        self.add_algorithms()

    def add_algorithms(self):
        algo = CombinationAlgorithm()
        self.algorithms[algo.name] = algo

    def __getitem__(self, name):
        if name in self.algorithms:
            return self.algorithms[name]
        return None
