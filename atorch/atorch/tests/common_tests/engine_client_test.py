import pickle
import unittest
from unittest import mock

from atorch.auto.engine_client import EngineClient


class FakeEasyDLClient(object):
    def get_task(self):
        pass

    def report_task_result(self):
        pass


class EngineClientt(unittest.TestCase):
    def setUp(self):
        self.test_client = EngineClient()
        if self.test_client.client is None:
            fake_client = FakeEasyDLClient()
            setattr(self.test_client, "client", fake_client)

    def test_get_analyse_task(self):
        task_info = ["default", "self-defined"]
        task_id = 0
        task_type = "ANALYSE"
        process_mode = "ONE_PROCESS"
        self.test_client.client.get_task = mock.MagicMock(return_value=(task_id, task_type, process_mode, 0, task_info))
        task = self.test_client.get_task()
        self.assertEqual(task.id, task_id)
        self.assertEqual(task.type, task_type)
        self.assertEqual(task.process_mode, process_mode)
        self.assertEqual(task.analysis_method, task_info)

    def test_get_parallel_group_task(self):
        task_info = {
            "model_parallel_size": 3,
            "model_parallel_group": [[0, 1, 2], [3, 4, 5]],
        }
        task_id = 0
        task_type = "SETUP_PARALLEL_GROUP"
        process_mode = "ALL_PROCESS"
        self.test_client.client.get_task = mock.MagicMock(
            return_value=(task_id, task_type, process_mode, 0, pickle.dumps(task_info))
        )
        task = self.test_client.get_task()
        self.assertEqual(task.id, task_id)
        self.assertEqual(task.type, task_type)
        self.assertEqual(task.process_mode, process_mode)
        self.assertEqual(task.parallel_group_info, task_info)
        with self.assertRaises(AttributeError):
            task.analysis_method

        with self.assertRaises(AttributeError):
            task.strategy

    def test_get_strategy_task(self):
        task_info = {
            "pipeline_parallel_size": 3,
            "pipeline_parallel_group": [[0, 1, 2], [3, 4, 5]],
        }
        serialized_task_info = pickle.dumps(task_info)
        serialized_methods = [
            ("1F1B", serialized_task_info, True),
            ("bidirectional", serialized_task_info, False),
        ]
        original_methods = [
            ("1F1B", task_info, True),
            ("bidirectional", task_info, False),
        ]
        task_id = 0
        task_type = "TUNE"
        process_mode = "ALL_PROCESS"
        self.test_client.client.get_task = mock.MagicMock(
            return_value=(task_id, task_type, process_mode, 0, serialized_methods)
        )
        task = self.test_client.get_task()
        self.assertEqual(task.id, task_id)
        self.assertEqual(task.type, task_type)
        self.assertEqual(task.process_mode, process_mode)
        self.assertEqual(task.strategy.opt_list, original_methods)
