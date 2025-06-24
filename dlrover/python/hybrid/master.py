import ray

from dlrover.python.hybrid.config import JobConfig
from dlrover.python.hybrid.manager import HybridManager


@ray.remote
class HybridMaster:
    def __init__(self, config: JobConfig):
        self.manager = HybridManager(config)

    def status(self):
        return self.manager.stage

    def start(self):
        self.manager.prepare()
        self.manager.start()

    def stop(self):
        self.manager.stop()

    # region RPC

    # endregion
