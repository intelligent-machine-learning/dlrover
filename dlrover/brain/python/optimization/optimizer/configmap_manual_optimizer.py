import time

from dlrover.brain.python.common.job import (
    JobOptimizePlan,
    JobMeta,
    JobResource,
    NodeGroupResource,
    OptimizeConfig,
)
from dlrover.brain.python.common.constants import (
    Node,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.brain.python.optimization.optalgorithm.opt_algorithm_manager import OptAlgorithmManager


class ConfigMapManualOptimizer:
    def __init__(self):
        self.opt_algorithm_manager = OptAlgorithmManager()

    @staticmethod
    def get_name() -> str:
        return "ConfigMapManualOptimizer"

    def optimize(self, job: JobMeta, conf: OptimizeConfig) -> JobOptimizePlan:
        conf.customized_config["algorithm"] = "ManualOptimizeJobResource"
        current_node_resource = self.opt_algorithm_manager.generate_node_resource(job, conf)
        return JobOptimizePlan(
            time=int(time.time() * 1000),
            job_resource=JobResource(
                node_group_resources={
                    Node.NODE_TYPE_WORKER: NodeGroupResource(
                        resource=current_node_resource,
                    )
                },
            ),
            job_meta=job,
        )