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
from unittest.mock import MagicMock, patch
from dlrover.brain.python.optimization.optimizer_manager import (
    OptimizerManager,
)
from dlrover.brain.python.optimization.optimizer.base_optimizer import (
    BaseOptimizer,
)
from dlrover.brain.python.common.job import (
    JobMeta,
    OptimizeConfig,
)
from dlrover.brain.python.common.constants import (
    Node,
    DefaultResource,
)


class TestOptimizerManager(unittest.TestCase):
    def setUp(self):
        self.optimizer_name = BaseOptimizer.get_name()
        self.manager = OptimizerManager()

    def test_optimize(self):
        """Scenario: Config requests 'my_algo', and it returns a valid plan."""
        job = JobMeta(
            uuid="job_uuid",
        )
        conf = OptimizeConfig(optimizer_name=self.optimizer_name)
        plan = self.manager.optimize(job, conf)
        job_resource = plan.job_resource
        self.assertEqual(
            job_resource.node_group_resources[
                Node.NODE_TYPE_WORKER
            ].resource.cpu,
            DefaultResource.WORKER_CPU,
        )

        conf.optimizer_name = "unknown"
        plan = self.manager.optimize(job, conf)
        self.assertIsNone(plan)

    @patch("logging.Logger.warning")  # Assuming you use standard logging
    def test_optimize_handles_exception(self, mock_log_warning):
        job = JobMeta(
            uuid="job_uuid",
        )
        conf = OptimizeConfig(optimizer_name="my_algo")

        mock_optimizer = MagicMock(spec=BaseOptimizer)
        mock_optimizer.optimize.side_effect = ValueError(
            "Something went wrong!"
        )
        self.manager.optimizers["my_algo"] = mock_optimizer

        # 3. Execution
        result = self.manager.optimize(job, conf)

        # 4. Assertions
        self.assertIsNone(result)

        expected_msg = "Fail to optimize job_uuid: Something went wrong!"
        mock_log_warning.assert_called_once_with(expected_msg)


if __name__ == "__main__":
    unittest.main()
