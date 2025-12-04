from typing import Dict, Optional

from dlrover.brain.python.common.optimize import (
    JobMeta,
    OptimizeConfig,
    JobOptimizePlan,
)
from dlrover.brain.python.optimizer.optimizer.base_optimizer import BaseOptimizer
from dlrover.brain.python.common.log import default_logger as logger


class OptimizerManager:
    def __init__(self):
        self.optimizers: Dict[str, BaseOptimizer] = {}

    def optimize(self, job: JobMeta, conf: OptimizeConfig) -> Optional[JobOptimizePlan, None]:
        if conf.optimizer not in self.optimizers:
            return None

        try:
            plan = self.optimizers[conf.optimizer].optimize(job)
            return plan
        except Exception as e:
            logger.warning(f"Fail to optimize job: {job}: {e}")
            return None
