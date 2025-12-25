import unittest
from unittest.mock import MagicMock, patch
from dlrover.brain.python.optimizer.optimizer_manager import OptimizerManager
from dlrover.brain.python.optimizer.optimizer.base_optimizer import BaseOptimizer
from dlrover.brain.python.common.optimize import (
    JobMeta,
    OptimizeConfig,
    JobOptimizePlan,
)


class TestOptimizerManager(unittest.TestCase):

    def setUp(self):
        self.optimizer_name = "test_optimizer"
        self.manager = OptimizerManager()
        self.mock_optimizer = MagicMock(spec=BaseOptimizer)
        self.manager.optimizers[self.optimizer_name] = self.mock_optimizer

    def test_optimize_success(self):
        """Scenario: Config requests 'my_algo', and it returns a valid plan."""
        job = MagicMock(spec=JobMeta)
        job_config = OptimizeConfig(optimizer_name="my_algo")
        plan = MagicMock(spec=JobOptimizePlan)

        # 2. Configure Mock Behavior
        self.mock_optimizer.optimize.return_value = fake_plan

        # 3. Execution
        result = self.manager.optimize(fake_job, fake_config)

        # 4. Assertions
        # Verify the manager called the correct optimizer with the job
        self.mock_optimizer.optimize.assert_called_once_with(fake_job)
        # Verify the result is what the optimizer returned
        self.assertEqual(result, fake_plan)

    def test_optimize_unknown_optimizer(self):
        """Scenario: Config requests 'unknown_algo' which is not in the dict."""
        # 1. Setup Input
        fake_job = MagicMock(spec=JobMeta)
        fake_config = OptimizeConfig(optimizer_name="unknown_algo")

        # 2. Execution
        result = self.manager.optimize(fake_job, fake_config)

        # 3. Assertions
        self.assertIsNone(result)
        # Verify our existing optimizer was NEVER touched
        self.mock_optimizer.optimize.assert_not_called()

    @patch("logging.Logger.warning") # Assuming you use standard logging
    def test_optimize_handles_exception(self, mock_log_warning):
        """Scenario: The optimizer crashes (raises Exception). Manager should catch it and return None."""
        # 1. Setup Input
        fake_job = MagicMock(spec=JobMeta)
        fake_config = OptimizeConfig(optimizer_name="my_algo")

        # 2. Configure Mock to Crash
        self.mock_optimizer.optimize.side_effect = ValueError("Something went wrong!")

        # 3. Execution
        result = self.manager.optimize(fake_job, fake_config)

        # 4. Assertions
        self.assertIsNone(result)
        # Verify the crash was actually triggered
        self.mock_optimizer.optimize.assert_called_once()
        # Verify the exception didn't crash the whole test (handled internally)

if __name__ == '__main__':
    unittest.main()