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

from typing import Dict, Optional

from dlrover.brain.python.common.job import (
    JobMeta,
    OptimizeConfig,
    JobOptimizePlan,
)
from dlrover.brain.python.optimization.optimizer.base_optimizer import (
    BaseOptimizer,
)
from dlrover.brain.python.common.log import default_logger as logger
from dlrover.brain.python.optimization.optimizer_router import OptimizerRouter


class OptimizerManager:
    """
    OptimizerManager manages registered optimizers. For an optimize request from
    the training job, it picks up one optimizer to optimize this job based runtime status of this job.
    """

    def __init__(self):
        self.optimizers: Dict[str, BaseOptimizer] = {}
        self.router = OptimizerRouter()

        self.register_optimizers()

    def register_optimizers(self):
        self.optimizers[BaseOptimizer.get_name()] = BaseOptimizer()

    def optimize(
        self, job: JobMeta, conf: OptimizeConfig
    ) -> Optional[JobOptimizePlan]:
        optimizer_name = self.router.route(job, conf)

        if optimizer_name not in self.optimizers:
            return None

        try:
            plan = self.optimizers[conf.optimizer_name].optimize(job)
            return plan
        except Exception as e:
            logger.warning(f"Fail to optimize {job.uuid}: {e}")
            return None
