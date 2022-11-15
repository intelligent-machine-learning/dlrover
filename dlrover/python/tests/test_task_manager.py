
import unittest
import json

from dlrover.python.master.shard_manager.task_manager import (
    TaskManager, DatasetShardCheckpoint
)
from dlrover.proto import elastic_training_pb2


def _create_task_manager():
    task_manager = TaskManager(False)
    dataset_name = "test"
    task_manager.new_dataset(
        batch_size=10,
        num_epochs=1,
        dataset_size=1000,
        shuffle=False,
        num_minibatches_per_shard=10,
        dataset_name=dataset_name,
        task_type=elastic_training_pb2.TRAINING,
        storage_type="table",
    )
    return task_manager
    

class TaskMangerTest(unittest.TestCase):
    def test_dispatch_task(self):
        dataset_name = "test"
        task_manager = _create_task_manager()
        self.assertEqual(len(task_manager._datasets), 1)
        task = task_manager.get_dataset_task(0, dataset_name)
        self.assertEqual(task.task_id, 0)
        dataset_manager = task_manager.get_dataset(dataset_name)
        self.assertIsNotNone(dataset_manager)

        request = elastic_training_pb2.ReportTaskResultRequest()
        request.task_id = 0
        request.dataset_name = dataset_name
        task, worker_id = task_manager.report_dataset_task(request, True)
        self.assertEqual(worker_id, 0)
        self.assertEqual(task.task_id, 0)
        self.assertGreater(task_manager._worker_start_task_time[0], 0)
        self.assertFalse(task_manager.finished())
        task = task_manager.get_dataset_task(dataset_name, 1)
        epoch = task_manager.get_dataset_epoch(dataset_name)
        self.assertEqual(epoch, 1)

    def test_recover_task(self):
        task_manager = _create_task_manager()
        dataset_name = "test"
        dataset = task_manager.get_dataset(dataset_name)
        task = task_manager.get_dataset_task(0, dataset_name)
        self.assertEqual(len(dataset.todo), 9)
        request = elastic_training_pb2.ReportTaskResultRequest()
        request.task_id = task.task_id
        request.dataset_name = dataset_name
        task_manager.report_dataset_task(request, False)
        self.assertEqual(len(dataset.todo), 10)
        self.assertEqual(len(dataset.doing), 0)

    def test_dataset_checkpoint(self):
        task_manager = _create_task_manager()
        dataset_name = "test"
        task_manager.get_dataset_task(0, dataset_name)
        task_manager.get_dataset_task(0, dataset_name)
        checkpoint: DatasetShardCheckpoint = task_manager.get_dataset_checkpoint(dataset_name)
        self.assertEqual(checkpoint.dataset_name, dataset_name)
        self.assertListEqual(checkpoint.doing, [[0,100], [100, 200]])
        self.assertEqual(len(checkpoint.todo), 8)
        self.assertEqual(checkpoint.epoch, 1)
        checkpoint_str = checkpoint.to_json()

        checkpoint_dict = json.loads(checkpoint_str)
        self.assertDictEqual(checkpoint_dict, {"dataset_name": "test", "todo": [[200, 300], [300, 400], [400, 500], [500, 600], [600, 700], [700, 800], [800, 900], [900, 1000]], "doing": [[0, 100], [100, 200]], "epoch": 1})

        dataset = task_manager.get_dataset(dataset_name)
        task_manager.get_dataset_task(0, dataset_name)
        self.assertEqual(len(dataset.todo), 7)
        task_manager.restore_dataset_from_checkpoint(checkpoint_str)
        self.assertEqual(dataset.todo[1].shard.start, 100)
        self.assertEqual(len(dataset.todo), 10)
