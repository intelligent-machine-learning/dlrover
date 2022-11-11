from abc import ABCMeta, abstractmethod
from dlrover.python.master.shard_manager.dataset_splitter import Shard


class Task(object):
    """A task contains a shard and will be assigned to a worker.
    Attributes:
        task_id: int, the index of the task.
        task_type: str, task type may be "training", "evaluation"
            and "prediction".
        shard: a record shard.
    """
    def __init__(self, task_id, task_type, shard: Shard):
        self.task_id = task_id
        self.task_type = task_type
        self.shard = shard


class ShardManger(metaclass=ABCMeta):
    @abstractmethod
    def get_task(self, worker_id):
        """Return a task with a shard for the worker with worker_id"""
        pass

    @abstractmethod
    def recover_task(self, shard_id):
        """Recover a dispatched task if a worker fails"""
        pass

    @abstractmethod
    def report_task_status(self, shard_id):
        """The worker reports the status of the shard"""
        pass

    @abstractmethod
    def checkpoint(self):
        """Checkpoint uncompleted shards"""
        pass

    @abstractmethod
    def restore_checkpoint(self, checkpoint):
        """Restore uncompleted data shards from a checkpoint"""
        pass
