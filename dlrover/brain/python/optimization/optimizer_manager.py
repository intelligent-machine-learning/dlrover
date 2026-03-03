from typing import Dict, Optional

from dlrover.brain.python.common.job import (
    JobMeta,
    OptimizeConfig,
    JobOptimizePlan,
)
from dlrover.brain.python.optimization.optimizer.base_optimizer import BaseOptimizer
from dlrover.brain.python.common.log import default_logger as logger
from dlrover.brain.python.optimization.optimizer_router import OptimizerRouter


class OptimizerManager:
    def __init__(self):
        self.optimizers: Dict[str, BaseOptimizer] = {}
        self.router = OptimizerRouter()

        self.register_optimizers()

    def register_optimizers(self):
        self.optimizers[BaseOptimizer.get_name()] = BaseOptimizer()

    def optimize(self, job: JobMeta, conf: OptimizeConfig) -> Optional[JobOptimizePlan]:
        optimizer_name = self.router.route(job, conf)

        if optimizer_name not in self.optimizers:
            return None

        try:
            plan = self.optimizers[conf.optimizer].optimize(job)
            return plan
        except Exception as e:
            logger.warning(f"Fail to optimize {job.uuid}: {e}")
            return None
