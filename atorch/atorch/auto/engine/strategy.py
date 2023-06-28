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

import collections
import pickle

from atorch.auto.engine.task import TaskType


class StrategyStatus:
    """strategy status
    INIT: strategy created but not tuned
    TUNED: strategy tuned successfully (or this strategy does not need tuning)
    SUCCEED: dryrun succeed.
    FAILED: tune or dryrun failed
    """

    INIT = 0
    TUNED = 1
    SUCCEED = 2
    FAILED = 3


StrategyInfo = collections.namedtuple(
    "StrategyInfo",
    "strategy status dryrun_result process_mode",
)


def strategy_duplicate(s1, s2):
    """check if s1 and s2 are duplicate strategies.
    Duplicate means they have same optimization method.
    Also if "parallel_mode" is used, configs are also the same.
    """
    if len(s1) != len(s2):
        return False
    for (name1, config1, _) in s1:
        found = False
        for (name2, config2, _) in s2:
            if name1 == name2:
                if name1 != "parallel_mode":
                    found = True
                else:
                    found = config1 == config2
            if found:
                break
        if not found:
            return False
    return True


class StrategyInfoCollection(object):
    """
    All Strategy info, including tune results, dryrun results, status.
    Also supports for running_task to strategy mapping.
    self.strategies: a dict with stategy id as key, StrategyInfo as value.
    """

    def __init__(self, opt_method_lib):
        self.strategies = {}
        self.task_to_strategy_id_mapping = {}
        self.id_to_unfinished_task_mapping = {}
        self.num_strategy = 0
        self.baseline_id = None
        self.opt_method_lib = opt_method_lib

    def add_strategy(self, strategy, is_baseline=False, skip_duplicate=True):
        """add strategy, return True if added successfully"""
        valid, p_mode = self.opt_method_lib.validate_strategy(strategy)
        if not valid:
            return False
        if skip_duplicate and self.is_duplicate(strategy):
            return False
        status = StrategyStatus.TUNED
        for (_, _, tunable) in strategy:
            if tunable:
                status = StrategyStatus.INIT
                break
        self.strategies[self.num_strategy] = StrategyInfo(strategy, status, None, p_mode)
        if is_baseline:
            self.baseline_id = self.num_strategy
        self.num_strategy += 1
        return True

    def is_duplicate(self, strategy):
        for s_id in self.strategies:
            if strategy_duplicate(self.strategies[s_id].strategy, strategy):
                return True
        return False

    def get_baseline_strategy(self):
        if self.baseline_id is None:
            return None
        return self.strategies[self.baseline_id]

    def update_strategy(self, s_id, strategy=None, status=None, dryrun_result=None):
        info = self.strategies[s_id]
        new_info = StrategyInfo(
            strategy if strategy is not None else info.strategy,
            status if status is not None else info.status,
            dryrun_result if dryrun_result is not None else info.dryrun_result,
            info.process_mode,
        )
        self.strategies[s_id] = new_info

    def get_inactive_strategy(self, process_mode=None):
        """An inactive strategy is a strategy that is not finished (status is
           not SUCCEED or FAILED) and there are no unfinished tasks associated
           with it.
        Return the strategy id or None if not found.
        If process_mode is not None, this strategy should be with process_mode.
        """
        for s_id in self.strategies:
            if self.strategies[s_id].status in [
                StrategyStatus.SUCCEED,
                StrategyStatus.FAILED,
            ] or (s_id in self.id_to_unfinished_task_mapping and len(self.id_to_unfinished_task_mapping[s_id]) > 0):
                continue
            if process_mode is not None and process_mode != self.strategies[s_id].process_mode:
                continue
            return s_id
        return None

    def assign_strategy_to_task(self, s_id, task_id):
        self.task_to_strategy_id_mapping[task_id] = s_id
        if s_id not in self.id_to_unfinished_task_mapping:
            self.id_to_unfinished_task_mapping[s_id] = []
        self.id_to_unfinished_task_mapping[s_id].append(task_id)

    def __getitem__(self, s_id):
        return self.strategies[s_id]

    def task_done(self, task_id, task_type, task_status, result):
        if task_id not in self.task_to_strategy_id_mapping.keys():
            return
        s_id = self.task_to_strategy_id_mapping[task_id]
        if s_id in self.id_to_unfinished_task_mapping and task_id in self.id_to_unfinished_task_mapping[s_id]:
            self.id_to_unfinished_task_mapping[s_id].remove(task_id)
        if task_status is False:
            self.update_strategy(s_id, status=StrategyStatus.FAILED)
        else:
            if task_type == TaskType.TUNE:
                status = StrategyStatus.TUNED
                self.update_strategy(s_id, strategy=result, status=status)
            elif task_type == TaskType.DRYRUN:
                status = StrategyStatus.SUCCEED
                self.update_strategy(s_id, status=status, dryrun_result=result)

    def get_succeeded_strategies(self):
        succeeded_strategies = {}
        for s_id, strategy_info in self.strategies.items():
            if strategy_info.status == StrategyStatus.SUCCEED:
                succeeded_strategies[s_id] = strategy_info
        return succeeded_strategies

    def get_best_strategy(self):
        best_s_id = None
        for s_id in self.strategies:
            if self.strategies[s_id].status == StrategyStatus.SUCCEED and (
                best_s_id is None
                or self.strategies[best_s_id].dryrun_result["throughput"]
                < self.strategies[s_id].dryrun_result["throughput"]
            ):
                best_s_id = s_id
        return None if best_s_id is None else self.strategies[best_s_id].strategy

    def get_parallel_mode_from_strategy(self, s_id):
        mode = None
        strategy = self.strategies[s_id].strategy
        for (name, config, _) in strategy:
            if name == "parallel_mode":
                mode = pickle.loads(config)
                break
        return mode
