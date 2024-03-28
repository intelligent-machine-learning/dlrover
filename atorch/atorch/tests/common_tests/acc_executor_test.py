import os
import unittest

from atorch.auto.engine.acceleration_engine import AccelerationEngine
from atorch.auto.engine.task import TaskType

os.environ["BO_SG_MAX_IETR"] = "2"


def process_task(tasks, executor):
    for (task, process_id) in tasks:
        result = None
        if task.task_type == TaskType.ANALYSE:
            result = {"model_params_num": 10000, "model_params_mb": 40000}
        if task.task_type == TaskType.TUNE:
            result = task.task_info
        if task.task_type == TaskType.DRYRUN:
            result = {"throughput": min(task.task_id + 2.0, 2)}
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
