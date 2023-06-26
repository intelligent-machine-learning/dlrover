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

import unittest

from atorch.auto.engine.optimization_method import OptimizationMethodLibrary
from atorch.auto.engine.strategy import StrategyInfoCollection, StrategyStatus
from atorch.auto.engine.task import TaskType


class TestStrategy(unittest.TestCase):
    def test_strategy(self):
        opt_method_lib = OptimizationMethodLibrary()
        strategy_info = StrategyInfoCollection(opt_method_lib)
        # add strategies
        strategy = [("parallel_mode", None, False)]
        strategy_info.add_strategy(strategy)
        strategy = [
            ("parallel_mode", None, False),
            ("amp_native", None, False),
        ]
        strategy_info.add_strategy(strategy)
        strategy = [("zero1", None, False), ("amp_native", None, False)]
        strategy_info.add_strategy(strategy)
        self.assertTrue(strategy_info.get_best_strategy() is None)
        task_id = 0
        dryrun_count = 0
        best_throughput = 0
        best_s_id = -1
        while True:
            s_id = strategy_info.get_inactive_strategy()
            if s_id is None:
                break
            strategy_info.assign_strategy_to_task(s_id, task_id)
            task_status = True
            result = None
            s_status = strategy_info[s_id].status
            if s_status == StrategyStatus.INIT:
                task_type = TaskType.TUNE
                result = strategy_info[s_id].strategy
            elif s_status == StrategyStatus.TUNED:
                task_type = TaskType.DRYRUN
                if dryrun_count == 1:
                    task_status = False
                result = {"throughput": task_id + 1}
                if task_status and best_throughput < result["throughput"]:
                    best_throughput = result["throughput"]
                    best_s_id = s_id
                dryrun_count += 1
            strategy_info.task_done(task_id, task_type, task_status, result)
            task_id += 1
        best_s = strategy_info.get_best_strategy()
        self.assertTrue(best_s is not None)
        self.assertEqual(best_s, strategy_info[best_s_id].strategy)
        strategy = [("zero1", None, False), ("amp_native", None, False)]
        self.assertTrue(strategy_info.is_duplicate(strategy))
        strategy = [("zero2", None, False)]
        self.assertFalse(strategy_info.is_duplicate(strategy))
        num_s = strategy_info.num_strategy
        strategy = [("zero1", None, False), ("amp_native", None, False)]
        strategy_info.add_strategy(strategy)
        self.assertEqual(num_s, strategy_info.num_strategy)
        opt_method_lib.methods["zero2"].disabled = True
        strategy = [("zero2", None, False), ("amp_native", None, False)]
        strategy_info.add_strategy(strategy)
        self.assertEqual(num_s, strategy_info.num_strategy)
        opt_method_lib.methods["zero2"].disabled = False
        strategy_info.add_strategy(strategy)
        self.assertEqual(num_s + 1, strategy_info.num_strategy)
