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

import unittest
from dlrover.brain.python.optimization.optimizer_router import OptimizerRouter
from dlrover.brain.python.common.job import (
    JobMeta,
    OptimizeConfig,
)


class TestOptimizerRouter(unittest.TestCase):
    def setUp(self):
        self.opt_router = OptimizerRouter()

    def test_optimizer_router(self):
        optimizer = "test_optimizer"
        job = JobMeta()
        conf = OptimizeConfig(
            optimizer_name=optimizer,
        )

        routed_optimizer = self.opt_router.route(job, conf)
        self.assertEqual(routed_optimizer, optimizer)


if __name__ == "__main__":
    unittest.main()
