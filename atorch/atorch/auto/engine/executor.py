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

import threading
import time

from atorch.auto.engine.strategy import StrategyStatus
from atorch.auto.engine.task import Task, TaskProcessMode, TaskStatus, TaskType
from atorch.common.log_utils import default_logger as logger


class ProcessStatus(object):
    """
    The status of all processes. A task_id or None (idle).
    """

    def __init__(self, total_process):
        self.pstatus = {idx: None for idx in range(total_process)}

    def idle(self, process_id):
        return self.pstatus[process_id] is None

    def all_idle(self):
        for idx in self.pstatus:
            if self.pstatus[idx] is not None:
                return False
        return True

    def __len__(self):
        return len(self.pstatus)

    def __getitem__(self, idx):
        return self.pstatus[idx]

    def __setitem__(self, idx, value):
        self.pstatus[idx] = value


class Executor(object):
    def __init__(
        self,
        opt_method_lib,
        algo_lib,
        strategy_infos,
        analyser_result,
        planner,
        device_context=None,
        included_opts=None,
        time_limit=None,
        verbose=False,
    ):
        """
        opt_method_lib: optimization method library.
        algo_lib: strategy generation algorithm library.
        strategy_infos: StrategyInfoCollection
        analyser_result: stores analyse task results.
        planner: planner to prune opt methods, select algorithms.
        device_context: a dict for device context attributes, such as
                        node_num, nproc_per_node, gpu_num_per_node, etc.
        included_opts: if not None, a list of optimization method names
                       that this strategy should use.
        time_limit: if not None, the acceleration process mush finish within
                    this time limit (in seconds).
        verbose: if print more log
        """
        self.opt_method_lib = opt_method_lib
        self.algo_lib = algo_lib
        self.strategy_infos = strategy_infos
        self.analyser_result = analyser_result
        self.planner = planner
        self.device_context = device_context
        self.included_opts = included_opts
        self.time_limit = time_limit
        self.start_time = time.time()
        self.end_time = None
        self.terminate_task_id = None
        self._lock = threading.Lock()
        self.current_parallel_mode = None

        if self.device_context is None:
            self.device_context = {"node_num": 1, "nproc_per_node": 1}
        self.total_process = self.device_context["node_num"] * self.device_context["nproc_per_node"]
        self.process_status = ProcessStatus(self.total_process)
        self._task_count = 0
        self.tasks = {}  # dict,  with task_id as key
        self.current_assigning_task_id = None
        self.pending_task_ids = []
        self.unfinished_task_num = 0
        self.task_done_processes_num = 0
        self.in_planner_stage = True
        self.selected_algos = []
        self.cur_algo_index = 0
        self.generate_tasks_if_needed()
        self.verbose = verbose

    def add_tasks(self, tasks):
        if isinstance(tasks, Task):
            tasks = [tasks]
        for t in tasks:
            t.task_id = self._task_count
            self._task_count += 1
            self.tasks[t.task_id] = t
            self.pending_task_ids.append(t.task_id)
            self.unfinished_task_num += 1
            if t.strategy_id >= 0:
                self.strategy_infos.assign_strategy_to_task(t.strategy_id, t.task_id)
            if t.task_type == TaskType.FINISH or t.task_type == TaskType.FAIL:
                self.terminate_task_id = t.task_id

    @property
    def can_be_terminated(self):
        return (
            self.terminate_task_id is not None
            and len(self.tasks[self.terminate_task_id].process_assigned) == self.total_process
        )

    @property
    def task_count(self):
        return self._task_count

    def assign_task_to_process(self, task_id, process_id):
        if self.verbose:
            logger.info(f"Assign task {task_id} to process {process_id}")
        self.tasks[task_id].add_process(process_id)
        self.process_status[process_id] = task_id

    def process_result(self, task_id, process_id, status, result):
        # only set current process itself as `idle`
        self.process_status[process_id] = None
        current_task_type = self.tasks[task_id].task_type
        if self.tasks[task_id].process_mode == TaskProcessMode.ALL_PROCESS:
            self.task_done_processes_num += 1
            # only process 0 update result and status
            if process_id > 0:
                return
        self.tasks[task_id].status = TaskStatus.SUCCEEDED if status else TaskStatus.FAILED
        self.tasks[task_id].task_result = result
        if status and current_task_type == TaskType.ANALYSE:
            self.analyser_result.update(result)
        if current_task_type == TaskType.SETUP_PARALLEL_GROUP:
            self.current_parallel_mode = self.tasks[task_id].task_info
        self.strategy_infos.task_done(task_id, self.tasks[task_id].task_type, status, result)

    def generate_tasks_if_needed(self, last_reported_task_id=None):
        if self.terminate_task_id is None and len(self.pending_task_ids) == 0:
            tasks = None
            if self.in_planner_stage:
                is_done, tasks, new_strategy_num, algos = self.planner.plan()
                if is_done:
                    self.in_planner_stage = False
                    self.selected_algos = algos
                    self.generate_tasks_if_needed(last_reported_task_id)
            else:
                for index in range(self.cur_algo_index, len(self.selected_algos)):
                    is_done, tasks, new_strategy_num = self.algo_lib[self.selected_algos[index]].strategy_generate(self)
                    if is_done:
                        self.cur_algo_index += 1
                    if tasks or new_strategy_num > 0:
                        break
            s_id = self.strategy_infos.get_inactive_strategy()
            s_tasks = self.generate_tasks_from_strategy(s_id)
            if tasks is None:
                tasks = []
            if s_tasks:
                tasks.extend(s_tasks)
            if tasks:
                self.add_tasks(tasks)
            elif self.unfinished_task_num == 0:
                self.add_final_task()
                self.end_time = time.time()
                self.report_acceleration_metric()

    def generate_tasks_from_strategy(self, s_id):
        tasks = []
        if s_id is not None:
            # tune or dryrun this strategy
            p_mode = self.strategy_infos.get_parallel_mode_from_strategy(s_id)
            task = Task(
                TaskType.SETUP_PARALLEL_GROUP,
                p_mode,
                process_mode=TaskProcessMode.ALL_PROCESS,
            )
            tasks.append(task)
            if self.strategy_infos[s_id].status == StrategyStatus.INIT:
                task_type = TaskType.TUNE
                process_mode = self.strategy_infos[s_id].process_mode
            else:
                task_type = TaskType.DRYRUN
                process_mode = TaskProcessMode.ALL_PROCESS
            task = Task(
                task_type,
                self.strategy_infos[s_id].strategy,
                process_mode=process_mode,
                strategy_id=s_id,
            )
            tasks.append(task)
        return tasks

    def add_final_task(self):
        best_strategy = self.strategy_infos.get_best_strategy()
        if best_strategy is None:
            task = Task(
                TaskType.FAIL,
                None,
                process_mode=TaskProcessMode.ALL_PROCESS,
            )
        else:
            task = Task(
                TaskType.FINISH,
                best_strategy,
                process_mode=TaskProcessMode.ALL_PROCESS,
            )
        self.add_tasks(task)

    def report_task_result(self, task_id, process_id, status, result):
        with self._lock:
            self.process_result(task_id, process_id, status, result)
            if self.tasks[task_id].process_mode == TaskProcessMode.ALL_PROCESS:
                if self.task_done_processes_num < self.total_process:
                    return
                self.task_done_processes_num = 0
            self.unfinished_task_num -= 1
            self.generate_tasks_if_needed(last_reported_task_id=task_id)

    def get_one_pending_task(self):
        t_id = None
        while len(self.pending_task_ids) > 0:
            t_id = self.pending_task_ids[0]
            self.pending_task_ids.remove(t_id)
            # check if task can be skipped
            if (
                self.tasks[t_id].task_type == TaskType.SETUP_PARALLEL_GROUP
                and self.tasks[t_id].task_info == self.current_parallel_mode
            ):
                t_id = None
                self.unfinished_task_num -= 1
            if t_id is not None:
                break
        return t_id

    def get_task(self, process_id):
        task = None
        with self._lock:
            if self.current_assigning_task_id is None and len(self.pending_task_ids) > 0:
                t_id = self.get_one_pending_task()
                if t_id is not None:
                    if self.tasks[t_id].process_mode == "ONE_PROCESS" or self.total_process == 1:
                        self.tasks[t_id].task_status = TaskStatus.RUNNING
                        task = self.tasks[t_id]
                        self.assign_task_to_process(t_id, process_id)
                    else:
                        self.current_assigning_task_id = t_id
                        self.tasks[t_id].task_status = TaskStatus.ASSIGNING
            if self.current_assigning_task_id is not None:
                # for ALL_PROCESS task, it must wait for all processes idle,
                # then assigning it to all processs.
                t_id = self.current_assigning_task_id
                if (
                    len(self.tasks[t_id].process_assigned) > 0 or self.process_status.all_idle()
                ) and process_id not in self.tasks[t_id].process_assigned:
                    # assign the task
                    self.assign_task_to_process(t_id, process_id)
                    task = self.tasks[t_id]
                if len(self.tasks[t_id].process_assigned) == self.total_process:
                    # All processes are assigned, start running
                    self.current_assigning_task_id = None
                    self.tasks[t_id].task_status = TaskStatus.RUNNING
        if task is None:
            if self.verbose:
                logger.info("Get None task, return WAIT task")
            task = Task(TaskType.WAIT, None)
        if self.verbose:
            logger.info(f"Process {process_id} get task, type {task.task_type}")
        return task

    def report_acceleration_metric(self):
        # TODO: report metrics to dlrover
        pass
