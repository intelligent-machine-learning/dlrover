import unittest
from unittest.mock import MagicMock, patch
from dlrover.brain.python.optimizer.optimizer_manager import OptimizerManager
from dlrover.brain.python.optimizer.optimizer.base_optimizer import BaseOptimizer
from dlrover.brain.python.common.optimize import (
    JobMeta,
    OptimizeConfig,
)
from dlrover.brain.python.common.constants import (
    Node,
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
        conf = OptimizeConfig(optimizer=self.optimizer_name)
        plan = self.manager.optimize(job, conf)
        job_resource = plan.job_resource
        self.assertEqual(job_resource.node_group_resources[Node.NODE_TYPE_WORKER].count, 4)

        conf.optimizer = "unknown"
        plan = self.manager.optimize(job, conf)
        self.assertIsNone(plan)

    @patch("logging.Logger.warning")  # Assuming you use standard logging
    def test_optimize_handles_exception(self, mock_log_warning):
        job = JobMeta(
            uuid="job_uuid",
        )
        conf = OptimizeConfig(optimizer="my_algo")

        mock_optimizer = MagicMock(spec=BaseOptimizer)
        mock_optimizer.optimize.side_effect = ValueError("Something went wrong!")
        self.manager.optimizers["my_algo"] = mock_optimizer

        # 3. Execution
        result = self.manager.optimize(job, conf)

        # 4. Assertions
        self.assertIsNone(result)

        expected_msg = "Fail to optimize job_uuid: Something went wrong!"
        mock_log_warning.assert_called_once_with(expected_msg)


if __name__ == '__main__':
    unittest.main()
