from dlrover.python.hybrid.defines import ActorBase, SubMaster
from dlrover.python.hybrid.elastic.manager import ElasticManager


class ElasticMaster(SubMaster):
    def __init__(self, master_config):
        self.manager = ElasticManager(master_config)

    def status(self):
        print("Elastic Master is running")

    def self_check(self):
        pass

    def check_workers(self):
        pass

    def start(self):
        self.manager.start()

    def stop(self):
        self.manager.stop()
