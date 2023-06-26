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

from atorch.auto.engine.acceleration_engine import AccelerationEngine
from atorch.auto.engine.task import TaskType


def process_task(tasks, executor):
    for (task, process_id) in tasks:
        result = None
        if task.task_type == TaskType.ANALYSE:
            result = {"model_params_num": 10000, "model_params_mb": 40000}
        if task.task_type == TaskType.TUNE:
            result = task.task_info
        if task.task_type == TaskType.DRYRUN:
            result = {"throughput": task.task_id + 2.0}
        if task.task_type != TaskType.WAIT:
            executor.report_task_result(task.task_id, process_id, True, result)


class TestExecutor(unittest.TestCase):
    def test_executor(self):
        device_context = {"node_num": 1, "nproc_per_node": 2}

        executor = AccelerationEngine.create_executor(device_context=device_context)

        process_running = [True for _ in range(2)]
        while any(process_running):
            tasks = []
            for idx, status in enumerate(process_running):
                if status:
                    task = executor.get_task(idx)
                    tasks.append((task, idx))
                    if task.task_type == TaskType.FINISH or task.task_type == TaskType.FAIL:
                        self.assertTrue(task.task_type != TaskType.FAIL)
                        process_running[idx] = False
            process_task(tasks, executor)
        self.assertTrue(executor.strategy_infos.num_strategy > 1)
