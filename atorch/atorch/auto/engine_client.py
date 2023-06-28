import pickle

from atorch.auto.engine.client import build_auto_acc_client
from atorch.auto.strategy import Strategy
from atorch.auto.task import Task
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import rank


class EngineClient(object):
    def __init__(self, engine_addr="localhost", engine_port="12345"):
        addr = "{}:{}".format(engine_addr, engine_port)
        self.process_id = rank()
        self.client = self.create_client(addr)

    def create_client(self, addr):
        """Return a AutoAccelerationClient defined in easydl."""
        return build_auto_acc_client(addr, self.process_id) if build_auto_acc_client is not None else None

    def get_task(self):
        """Get task from easydl."""
        if self.client is None:
            logger.warning("AutoAccelerationClient is None. Cannot call `get_task`. Just return.")
            return
        task_id, task_type, process_mode, time_limit, task_info = self.client.get_task()
        deserialized_task_info = None
        if task_type in ["TUNE", "DRYRUN", "FINISH"]:
            # deserialized_task_info: List[Tuple[str, Dict, bool]]
            deserialized_task_info = []
            for info in task_info:
                deserialized_task_info.append((info[0], pickle.loads(info[1]), info[2]))
            deserialized_task_info = Strategy(deserialized_task_info)
        elif task_type == "SETUP_PARALLEL_GROUP":
            # deserialized_task_info: Dict
            deserialized_task_info = pickle.loads(task_info)
        elif task_type == "ANALYSE":
            # deserialized_task_info: List[str]
            deserialized_task_info = task_info
        task = Task(id=task_id, task_type=task_type, process_mode=process_mode, task_info=deserialized_task_info)
        return task

    def report_task_result(self, task, status, result):
        """Report task result."""
        if self.client is None:
            logger.warning("AutoAccelerationClient is None. Cannot call `report_task_result`. Just return.")
            return
        task_id = task.id
        task_type = task.type
        if task_type == "TUNE" and status:
            serialized_result = []
            for method in result:
                serialized_result.append((method[0], pickle.dumps(method[1]), method[2]))
        else:
            serialized_result = pickle.dumps(result)

        self.client.report_task_result(task_id, task_type, status, serialized_result)
