from dlrover.brain.python.common.job import (
    JobMeta,
    OptimizeConfig,
)


class OptimizerRouter:
    def __init__(self):
        pass

    def route(self, job: JobMeta, conf: OptimizeConfig) -> str:
        return conf.optimizer
