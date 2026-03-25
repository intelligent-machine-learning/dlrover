# Copyright 2026 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

from dlrover.brain.python.common.job import (
    JobOptimizePlan,
    OptimizeConfig,
    JobMeta,
)
from pydantic import BaseModel, Field


class Response(BaseModel):
    success: bool = False
    reason: str = ""


class OptimizeResponse(BaseModel):
    response: Response = Field(default_factory=Response)
    job_opt_plan: JobOptimizePlan = Field(default_factory=JobOptimizePlan)


class OptimizeRequest(BaseModel):
    type: str = ""
    config: OptimizeConfig = Field(default_factory=OptimizeConfig)
    job: JobMeta = Field(default_factory=JobMeta)


class JobConfigRequest(BaseModel):
    type: str = ""
    job: JobMeta = Field(default_factory=JobMeta)


class JobConfigResponse(BaseModel):
    response: Response = Field(default_factory=Response)
    job_configs: Dict[str, str] = Field(default_factory=dict)
