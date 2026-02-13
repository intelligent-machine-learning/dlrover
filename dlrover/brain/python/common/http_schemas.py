from dataclasses import field
from typing import Dict

from dlrover.brain.python.common.job import (
    JobOptimizePlan,
    OptimizeConfig,
    JobMeta,
)
from pydantic import BaseModel


class Response(BaseModel):
    success: bool = False
    reason: str = ""


class OptimizeResponse(BaseModel):
    response: Response = field(default_factory=Response)
    job_opt_plan: JobOptimizePlan = field(default_factory=JobOptimizePlan)


class OptimizeRequest(BaseModel):
    type: str = ""
    config: OptimizeConfig = field(default_factory=OptimizeConfig)
    job: JobMeta = field(default_factory=JobMeta)


class JobConfigRequest(BaseModel):
    type: str = ""
    job: JobMeta = field(default_factory=JobMeta)


class JobConfigResponse(BaseModel):
    response: Response = field(default_factory=Response)
    job_configs: Dict[str, str] = field(default_factory=dict)
