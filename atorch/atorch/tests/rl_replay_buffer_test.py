import unittest

import torch

from atorch.rl.config import AtorchRLConfig
from atorch.rl.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_replay_buffer(self):
        config_path = "./atorch/tests/test_define_rl_models/independent_models/model_def.yaml"
        atorch_rl_config = AtorchRLConfig.load_yaml(config_path)
        atorch_rl_config.tokenizer.pad_token_id = 50005
        replay_buffer = ReplayBuffer(config=atorch_rl_config)
        replay_buffer.reset()
        batch_size = 4
        sequence_length = 20
        embedding_size = 2048
        query_tensor = torch.rand((batch_size, sequence_length))
        response_tensor = torch.rand((batch_size, sequence_length))
        logprobs = torch.rand((batch_size, sequence_length, embedding_size))
        values = torch.rand((batch_size, sequence_length))
        rewards = torch.rand((batch_size, sequence_length))
        ppo_element = {
            "query_tensor": query_tensor,
            "response_tensor": response_tensor,
            "logprobs": logprobs,
            "values": values,
            "rewards": rewards,
        }
        replay_buffer.add_sample(ppo_element)
        self.assertEqual(replay_buffer.num, 1)
        replay_buffer.add_samples([ppo_element])
        self.assertEqual(replay_buffer.num, 2)
        replay_buffer.add_sample(ppo_element, index=2)
        self.assertEqual(replay_buffer.num, 2)
        new_query_tensor = torch.rand(1)
        new_ppo_element = {
            "query_tensor": new_query_tensor,
            "response_tensor": response_tensor,
            "logprobs": logprobs,
            "values": values,
            "rewards": rewards,
        }
        index = 1
        replay_buffer.add_sample(new_ppo_element, index=index)
        self.assertEqual(replay_buffer.data["query_tensor"][index].shape[0], 1)
        rl_training_dataset = replay_buffer.create_dataset()
        self.assertEqual(len(rl_training_dataset), 2)


if __name__ == "__main__":
    unittest.main()
