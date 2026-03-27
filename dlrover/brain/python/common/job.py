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

from pydantic import (
    Field,
    BaseModel,
)
from typing import Dict


class NodeResource(BaseModel):
    memory: int = 0
    cpu: float = 0.0  # The number of cores
    gpu: float = 0.0
    gpu_type: str = ""
    priority: int = 0


class NodeGroupResource(BaseModel):
    count: int = 0
    resource: NodeResource = NodeResource()


class JobResource(BaseModel):
    node_group_resources: Dict[str, NodeGroupResource] = Field(
        default_factory=dict
    )
    node_resources: Dict[str, NodeResource] = Field(default_factory=dict)


class JobMeta(BaseModel):
    uuid: str = ""
    cluster: str = ""
    namespace: str = ""
    user: str = ""
    app: str = ""


class JobOptimizePlan(BaseModel):
    timestamp: int = 0
    job_resource: JobResource = JobResource()
    job_meta: JobMeta = JobMeta()


class OptimizeConfig(BaseModel):
    optimizer_name: str = ""
    customized_config: Dict[str, str] = Field(default_factory=dict)
