import unittest

import torch

from atorch.common.log_utils import DashBoardWriter
from atorch.rl.config import AtorchRLConfig
from atorch.rl.ppo_utils.ppo_util import get_advantages_and_returns, loss


class TestRLTrainerUtil(unittest.TestCase):
    def test_rl_trainer_util_cpu(self):
        # TODO: test whether the result is right.
        config_path = "./atorch/tests/test_define_rl_models/independent_models/model_def.yaml"
        atorch_rl_config = AtorchRLConfig.load_yaml(config_path)

        batch_size = 2
        sequence_length = 20
        embedding_size = 2048
        response_tensor = torch.ones((batch_size, sequence_length))
        logits = torch.ones((batch_size, sequence_length, embedding_size))
        logprobs = torch.ones((batch_size, sequence_length, embedding_size))
        values = torch.ones((batch_size, sequence_length))
        rewards = torch.ones((batch_size, sequence_length))

        response_length = response_tensor.shape[1]

        advantages, returns = get_advantages_and_returns(
            values, rewards, response_length, config=atorch_rl_config.ppo_config
        )
        self.assertEqual(advantages.dim(), returns.dim())

        logprobs = torch.tensor(
            [
                [-1.3162, -4.4145, -2.9761, -1.5285, -2.4031, -0.7153, -1.3419, -0.9945],
                [-1.3162, -4.4145, -2.9761, -1.5285, -2.4031, -0.7153, -1.3419, -0.9945],
            ]
        )

        values = torch.tensor(
            [
                [-0.2439, -0.2515, -0.0448, -0.0810, -0.1800, -0.3194, -0.3549, -0.3704],
                [-0.2439, -0.2515, -0.0448, -0.0810, -0.1800, -0.3194, -0.3549, -0.3704],
            ]
        )
        old_logprobs = torch.tensor(
            [
                [-1.3162, -4.4145, -2.9761, -1.5285, -2.4031, -0.7153, -1.3419, -0.9945],
                [-1.3162, -4.4145, -2.9761, -1.5285, -2.4031, -0.7153, -1.3419, -0.9945],
            ]
        )
        old_values = torch.tensor(
            [
                [-0.2439, -0.2515, -0.0448, -0.0810, -0.1800, -0.3194, -0.3549, -0.3704],
                [-0.2439, -0.2515, -0.0448, -0.0810, -0.1800, -0.3194, -0.3549, -0.3704],
            ]
        )
        advantages = torch.tensor(
            [
                [0.2212, 0.2474, 0.0373, 0.0645, 0.1509, 0.2872, 0.3268, 0.8936],
                [0.2212, 0.2474, 0.0373, 0.0645, 0.1509, 0.2872, 0.3268, 0.8936],
            ]
        )
        returns = torch.tensor(
            [
                [-0.0227, -0.0041, -0.0075, -0.0165, -0.0291, -0.0322, -0.0281, 0.5232],
                [-0.0227, -0.0041, -0.0075, -0.0165, -0.0291, -0.0322, -0.0281, 0.5232],
            ]
        )
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]])
        logits = torch.ones((2, mask.shape[1], embedding_size))
        ppo_loss, stats = loss(
            logprobs,
            values,
            old_logprobs,
            old_values,
            advantages,
            returns,
            mask,
            logits,
            config=atorch_rl_config.ppo_config,
        )
        dashboard_writer = DashBoardWriter("./")
        dashboard_writer.add_scalars(stats, 1)
        self.assertEqual(ppo_loss.dim(), 0)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_rl_trainer_util_gpu(self):
        # TODO: test whether the result is right.
        config_path = "./atorch/tests/test_define_rl_models/independent_models/model_def.yaml"
        atorch_rl_config = AtorchRLConfig.load_yaml(config_path)

        batch_size = 2
        sequence_length = 20
        embedding_size = 2048
        response_tensor = torch.ones((batch_size, sequence_length)).cuda()
        logits = torch.ones((batch_size, sequence_length, embedding_size)).cuda()
        logprobs = torch.ones((batch_size, sequence_length, embedding_size)).cuda()
        values = torch.ones((batch_size, sequence_length)).cuda()
        rewards = torch.ones((batch_size, sequence_length)).cuda()

        response_length = response_tensor.shape[1]

        advantages, returns = get_advantages_and_returns(
            values, rewards, response_length, config=atorch_rl_config.ppo_config
        )
        self.assertEqual(advantages.dim(), returns.dim())

        logprobs = torch.tensor(
            [
                [-1.3162, -4.4145, -2.9761, -1.5285, -2.4031, -0.7153, -1.3419, -0.9945],
                [-1.3162, -4.4145, -2.9761, -1.5285, -2.4031, -0.7153, -1.3419, -0.9945],
            ]
        ).cuda()

        values = torch.tensor(
            [
                [-0.2439, -0.2515, -0.0448, -0.0810, -0.1800, -0.3194, -0.3549, -0.3704],
                [-0.2439, -0.2515, -0.0448, -0.0810, -0.1800, -0.3194, -0.3549, -0.3704],
            ]
        ).cuda()
        old_logprobs = torch.tensor(
            [
                [-1.3162, -4.4145, -2.9761, -1.5285, -2.4031, -0.7153, -1.3419, -0.9945],
                [-1.3162, -4.4145, -2.9761, -1.5285, -2.4031, -0.7153, -1.3419, -0.9945],
            ]
        ).cuda()
        old_values = torch.tensor(
            [
                [-0.2439, -0.2515, -0.0448, -0.0810, -0.1800, -0.3194, -0.3549, -0.3704],
                [-0.2439, -0.2515, -0.0448, -0.0810, -0.1800, -0.3194, -0.3549, -0.3704],
            ]
        ).cuda()
        advantages = torch.tensor(
            [
                [0.2212, 0.2474, 0.0373, 0.0645, 0.1509, 0.2872, 0.3268, 0.8936],
                [0.2212, 0.2474, 0.0373, 0.0645, 0.1509, 0.2872, 0.3268, 0.8936],
            ]
        ).cuda()
        returns = torch.tensor(
            [
                [-0.0227, -0.0041, -0.0075, -0.0165, -0.0291, -0.0322, -0.0281, 0.5232],
                [-0.0227, -0.0041, -0.0075, -0.0165, -0.0291, -0.0322, -0.0281, 0.5232],
            ]
        ).cuda()
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]]).cuda()
        logits = torch.ones((batch_size, mask.shape[1], embedding_size)).cuda()
        ppo_loss, stats = loss(
            logprobs,
            values,
            old_logprobs,
            old_values,
            advantages,
            returns,
            mask,
            logits,
            config=atorch_rl_config.ppo_config,
        )
        self.assertEqual(ppo_loss.dim(), 0)
        dashboard_writer = DashBoardWriter("./")
        dashboard_writer.add_scalars(stats, 1)


if __name__ == "__main__":
    unittest.main()
