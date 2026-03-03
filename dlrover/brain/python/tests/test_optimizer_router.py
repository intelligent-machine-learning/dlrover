import unittest
from dlrover.brain.python.optimizer.optimizer_router import OptimizerRouter
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
            optimizer=optimizer,
        )

        routed_optimizer = self.opt_router.route(job, conf)
        self.assertEqual(routed_optimizer, optimizer)


if __name__ == '__main__':
    unittest.main()