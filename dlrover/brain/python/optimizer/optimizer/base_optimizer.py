from dlrover.brain.python.common.optimize import (
    JobOptimizePlan,
    JobMeta,
)


class BaseOptimizer:
    def __init__(self):
        pass

    def optimize(self, job: JobMeta) -> JobOptimizePlan:
        return JobOptimizePlan(
            
        )
