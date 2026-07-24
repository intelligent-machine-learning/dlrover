import threading

from typing import Dict

from dlrover.brain.python.common.job import (
    JobMeta,
    OptimizeConfig,
    NodeResource,
)
from dlrover.brain.python.common.log import default_logger as logger

from dlrover.brain.python.optimization.optalgorithm.manual_optimize_job_resource import ManualOptimizeJobResource
from dlrover.brain.python.optimization.optalgorithm.base_optimize_job_resource import BaseOptimizeJobResource

_locker = threading.RLock()

class OptAlgorithmManager:
    def __init__(self):
        self.opt_algorithm_library: Dict[str, BaseOptimizeJobResource] = {}
        self.current_node_resource = NodeResource()
        self.register_optAlgorithm()

    def register_optAlgorithm(self):
        self.opt_algorithm_library[BaseOptimizeJobResource.get_name()] = BaseOptimizeJobResource()
        self.opt_algorithm_library[ManualOptimizeJobResource.get_name()] = ManualOptimizeJobResource()

    def generate_node_resource(self, job: JobMeta, conf: OptimizeConfig) -> NodeResource:
        algorithm = conf.customized_config.get("algorithm", "")

        if algorithm not in self.opt_algorithm_library:
            logger.warning(f"Invalid algorithm config for job {job.uuid}")
            return self.current_node_resource

        try:
            with _locker:
                node_resource = self.opt_algorithm_library[algorithm].generate_node_resource(job, conf)
            return node_resource
        except Exception as e:
            logger.error(f"Fail to generate node resource {job.uuid}: {e}")
            return self.current_node_resource
