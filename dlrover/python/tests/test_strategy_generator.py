import unittest
from dlrover.python.master.hyperparams.simple_strategy_generator import SimpleStrategyGenerator
from dlrover.python.common.grpc import DataLoaderConfig, OptimizerConfig, ParallelConfig


class TestLocalStrategyGenerator(unittest.TestCase):
    def setUp(self):
        self.job_uuid = '1234'
        self._strategy_generator = SimpleStrategyGenerator(self.job_uuid)

    def test_generate_opt_strategy(self):
        gpu_stats = [
            {
                "index": 0,
                "total_memory_gb": 40,
                "used_memory_gb": 2,
            },
            {
                "index": 1,
                "total_memory_gb": 40,
                "used_memory_gb": 12,
            },
        ]

        model_config = {
            "block_size": 128,
            "n_layer": 6,
            "n_heads": 6,
            "n_embd": 384,
        }

        dataloader_config = DataLoaderConfig(0, "simple_dataloader", 32, 2, 0)
        optimizer_config = OptimizerConfig(0, 0)
        paral_config = ParallelConfig(dataloader_config, optimizer_config)

        expected_dataloader_config = DataLoaderConfig(
            1, "simple_dataloader", 1800, 0, 0)
        expected_optimizer_config = OptimizerConfig(5, 6)

        result = self._strategy_generator.generate_opt_strategy(
            gpu_stats, model_config)
        print(result)
        self.assertEqual(expected_dataloader_config, result.dataloader)
        self.assertEqual(expected_optimizer_config, result.optimizer)


if __name__ == '__main__':
    unittest.main()
