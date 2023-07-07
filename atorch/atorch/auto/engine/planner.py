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

from atorch.auto.engine.task import Task, TaskType


class PlannerStage:
    BASIC_PRUNE = 0
    ANALYSE = 1
    BASELINE_STRATEGY = 2
    SELECT_ALGO = 3


class Planner(object):
    """Planner analyses the model, dryruns baseline strategy,
    prunes the optimization methods, and select strategy generation algorithms.
    stage 0: basic prune
    stage 1: analyse model
    stage 2: dryrun with baseline strategy
    stage 3: advance prune and select algorithms
    """

    def __init__(
        self,
        opt_method_lib,
        algo_lib,
        strategy_infos,
        analyser_result,
        device_context,
        load_strategy=None,
        included_opts=None,
        excluded_opts=None,
    ):
        self.opt_method_lib = opt_method_lib
        self.algo_lib = algo_lib
        self.strategy_infos = strategy_infos
        self.analyser_result = analyser_result
        self.device_context = device_context or {}
        self.load_strategy = load_strategy
        self.included_opts = included_opts
        self.excluded_opts = excluded_opts or []
        self.selected_algo = None
        self.basic_prune()
        self.stage = PlannerStage.BASIC_PRUNE
        self.algos = []

    def basic_prune(self):
        # prune opt methods using excluded_opts and device context
        if not self.device_context:
            self.total_process = 1
        else:
            self.total_process = self.device_context["node_num"] * self.device_context["nproc_per_node"]
        gpu_available = self.device_context.get("total_gpu", 0) > 0
        gpu_compute_capability = float(self.device_context.get("gpu_compute_capability", "0.0"))

        for name, opt in self.opt_method_lib.methods.items():
            # prune based on sm_version(gpu_compute_capability)
            if opt.min_sm_version and gpu_compute_capability < float(opt.min_sm_version):
                self.excluded_opts.append(name)
            # If node has no GPU, prune opt that only supports GPU
            elif not gpu_available and "cpu" not in opt.supported_devices:
                self.excluded_opts.append(name)
            # If only has 1 process, prune opt that support distributed_only
            elif self.total_process == 1 and opt.distributed_only:
                self.excluded_opts.append(name)

        self.excluded_opts = list(set(self.excluded_opts))
        self.opt_method_lib.disable_opts(self.excluded_opts)

    def advance_prune(self):
        # prune opt methods using analyser result and baseline strategy results
        can_module_replace = self.analyser_result.get("has_module_for_replace")
        if not can_module_replace:
            self.excluded_opts.append("module_replace")
            self.opt_method_lib.disable_opts(["module_replace"])

    def generate_baseline_strategy(self):
        # create baseline strategy and add it to streategy_infos
        # empty for non-distributed, ddp only for distributed.
        if self.total_process > 1:
            p_mode = ([("data", self.total_process)], None)
            dp_strategy = [("parallel_mode", pickle.dumps(p_mode), False)]
            self.strategy_infos.add_strategy(dp_strategy, is_baseline=True)
        else:
            self.strategy_infos.add_strategy([], is_baseline=True)

    def generate_analyse_tasks(self):
        # generate analyse tasks
        task_info = ["analyse_basic"]
        task = Task(TaskType.ANALYSE, task_info)
        return [task]

    def select_algos(self):
        # select a list of algorithms
        # TODO: when multiple algorithms are added, need to select smartly.
        self.algos = ["combination_sg"]

    def plan(self):
        # Return is_done, tasks, new_strategy_num, algos
        is_done = False
        tasks = None
        new_strategy_num = 0
        if self.load_strategy is not None:
            self.strategy_infos.add_strategy(self.load_strategy)
            new_strategy_num = 1
            is_done = True
        elif self.stage == PlannerStage.BASIC_PRUNE:
            tasks = self.generate_analyse_tasks()
            self.stage = PlannerStage.ANALYSE
        elif self.stage == PlannerStage.ANALYSE:
            self.generate_baseline_strategy()
            new_strategy_num = 1
            self.stage = PlannerStage.BASELINE_STRATEGY
        elif self.stage == PlannerStage.BASELINE_STRATEGY:
            self.advance_prune()
            self.select_algos()
            self.stage = PlannerStage.SELECT_ALGO
            is_done = True
        else:
            is_done = True
        return is_done, tasks, new_strategy_num, self.algos
