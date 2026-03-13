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

import time

from dlrover.brain.python.common.job import (
    JobOptimizePlan,
    JobMeta,
    JobResource,
    NodeGroupResource,
    NodeResource,
)
from dlrover.brain.python.common.constants import (
    Node,
    DefaultResource,
)


class BaseOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def get_name() -> str:
        return "BaseOptimizer"

    def optimize(self, job: JobMeta) -> JobOptimizePlan:
        return JobOptimizePlan(
            timestamp=int(time.time() * 1000),
            job_resource=JobResource(
                node_group_resources={
                    Node.NODE_TYPE_WORKER: NodeGroupResource(
                        resource=NodeResource(
                            cpu=DefaultResource.WORKER_CPU,
                            memory=DefaultResource.WORKER_MEM,
                        ),
                    )
                },
            ),
            job_meta=job,
        )
