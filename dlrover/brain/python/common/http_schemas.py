from dlrover.brain.python.common.optimize import (
    JobOptimizePlan,
    OptimizeConfig,
    JobMeta,
)


class Response:
    success: bool
    reason: str


class OptimizeResponse:
    response: Response
    job_opt_plan: JobOptimizePlan


class OptimizeRequest:
    type: str
    config: OptimizeConfig
    job: JobMeta





