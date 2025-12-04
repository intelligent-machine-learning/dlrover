from dlrover.brain.python.common.optimize import (
    JobOptimizePlan,
    OptimizeConfig,
    JobMeta,
)
from pydantic import BaseModel


class Response(BaseModel):
    success: bool = False
    reason: str = ""


class OptimizeResponse(BaseModel):
    response: Response = Response()
    job_opt_plan: JobOptimizePlan = JobOptimizePlan()


class OptimizeRequest(BaseModel):
    type: str = ""
    config: OptimizeConfig = OptimizeConfig()
    job: JobMeta = JobMeta()





