import time

from dlrover.brain.python.common.optimize import (
    JobOptimizePlan,
    JobMeta,
    JobResource,
    NodeGroupResource,
    NodeResource,
)
from dlrover.brain.python.common.constants import (
    Node,
)


class BaseOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def get_name() -> str:
        return "BaseOptimizer"

    def optimize(self, job: JobMeta) -> JobOptimizePlan:
        return JobOptimizePlan(
            time=int(time.time() * 1000),
            job_resource=JobResource(
                node_group_resources={
                    Node.NODE_TYPE_WORKER: NodeGroupResource(
                        count=4,
                    )
                },
            ),
            job_meta=job,
        )
