import unittest

from atorch.rl.config import AtorchRLConfig, TrainableModelConfig
from atorch.utils.import_util import import_module


class TestAtorchRLConfig(unittest.TestCase):
    def test_rl_config(self):
        config_path = "./atorch/tests/test_define_rl_models/independent_models/model_def.yaml"
        atorch_rl_config = AtorchRLConfig.load_yaml(config_path)
        self.assertTrue(isinstance(atorch_rl_config.model.actor, TrainableModelConfig))
        self.assertEqual(
            atorch_rl_config.model.actor.model_path,
            "./atorch/tests/test_define_rl_models/independent_models/model_definition.py",
        )
        self.assertEqual(atorch_rl_config.model.actor.optimizer.name, "torch.optim.adam")
        self.assertEqual(
            atorch_rl_config.model.actor.train_strategy,
            "./atorch/tests/test_define_rl_models/independent_models/strategy.py",
        )
        self.assertDictEqual(atorch_rl_config.model.actor.model_params, {"features_in": 10, "features_out": 10})
        self.assertEqual(atorch_rl_config.model.actor.model_cls, "FakeActor")
        self.assertEqual(atorch_rl_config.model.critic.model_cls, "FakeCritic")
        self.assertEqual(atorch_rl_config.train.batch_size, 4)
        self.assertEqual(atorch_rl_config.train.num_rollouts, 10)
        self.assertTrue(isinstance(atorch_rl_config.model.actor.peft_config, dict))

    def test_rl_hg_config(self):
        config_path = "./atorch/tests/test_define_rl_models/independent_models/hg_model_def.yaml"
        atorch_rl_config = AtorchRLConfig.load_yaml(config_path)
        self.assertTrue(isinstance(atorch_rl_config.model.actor, TrainableModelConfig))
        self.assertEqual(atorch_rl_config.model.actor.model_path, "/home/glm-large-chinese")
        self.assertEqual(atorch_rl_config.model.actor.optimizer.name, "torch.optim.adam")
        self.assertDictEqual(atorch_rl_config.model.actor.model_params, {"features_in": 10, "features_out": 10})
        self.assertEqual(atorch_rl_config.model.actor.model_cls, "transformers.AutoModelForSeq2SeqLM")
        self.assertEqual(atorch_rl_config.model.actor.peft_config, None)
        self.assertEqual(atorch_rl_config.model.critic.model_cls, "transformers.AutoModelForSeq2SeqLM")
        self.assertEqual(atorch_rl_config.train.batch_size, 4)
        self.assertEqual(atorch_rl_config.train.num_rollouts, 4)
        self.assertEqual(atorch_rl_config.train.logdir, "./tensorboard")
        self.assertEqual(atorch_rl_config.train.num_rollouts, 4)
        self.assertDictEqual(
            atorch_rl_config.generation.gen_kwargs,
            {"max_new_tokens": 512, "top_k": 0, "top_p": 1.0, "do_sample": False},
        )
        self.assertDictEqual(
            atorch_rl_config.to_dict(),
            {
                "model": {
                    "model": {
                        "actor": {
                            "optimizer": {"name": "torch.optim.adam", "kwargs": {"betas": (0.9, 0.999)}},
                            "model_path": "/home/glm-large-chinese",
                            "model_cls": "transformers.AutoModelForSeq2SeqLM",
                            "model_params": {"features_in": 10, "features_out": 10},
                            "peft_config": None,
                            "train_strategy": "torch_native",
                            "inference_strategy": "torch_native",
                            "lazy_load": False,
                            "loss": "",
                        },
                        "critic": {
                            "optimizer": {"name": "torch.optim.adam", "kwargs": {"betas": (0.9, 0.999)}},
                            "model_path": "/home/glm-large-chinese",
                            "model_cls": "transformers.AutoModelForSeq2SeqLM",
                            "model_params": {"dims_in": 2, "dims_out": 2},
                            "peft_config": None,
                            "train_strategy": "torch_native",
                            "inference_strategy": "torch_native",
                            "lazy_load": False,
                            "loss": "",
                        },
                        "ref_model": {
                            "model_path": "/home/glm-large-chinese",
                            "model_cls": "transformers.AutoModelForSeq2SeqLM",
                            "model_params": {"dims_in": 2, "dims_out": 2},
                            "train_strategy": "torch_native",
                            "inference_strategy": "torch_native",
                            "lazy_load": False,
                            "peft_config": None,
                        },
                        "reward_model": {
                            "model_path": "/home/glm-large-chinese",
                            "model_cls": "transformers.AutoModelForSeq2SeqLM",
                            "model_params": {"dims_in": 2, "dims_out": 2},
                            "train_strategy": "torch_native",
                            "inference_strategy": "torch_native",
                            "lazy_load": False,
                            "peft_config": None,
                        },
                    }
                },
                "ppo_config": {
                    "ppo_epoch": 2,
                    "init_kl_coef": 0.02,
                    "gamma": 1,
                    "lam": 0.95,
                    "cliprange": 0.2,
                    "cliprange_value": 0.2,
                    "vf_coef": 0.1,
                    "cliprange_reward": 50,
                    "ent_coef": 0.01,
                    "horizon": 10000.0,
                    "clip_ratio": True,
                    "scale_reward": "running",
                    "ref_mean": None,
                    "ref_std": None,
                },
                "train": {
                    "seq_length": 1024,
                    "batch_size": 4,
                    "epoch": 1,
                    "num_rollouts": 4,
                    "mode": "Concurrent",
                    "trainer": "PPOTrainer",
                    "logdir": "./tensorboard",
                    "scheduler": {
                        "name": "cosine_warmup",
                        "kwargs": {"num_warmup_steps": 640, "num_training_steps": 6400},
                    },
                    "eval_interval": 100,
                    "checkpoint_interval": 100,
                    "checkpoint_dir": "./checkpoint",
                    "gradient_accumulation_steps": 1,
                    "max_grad_norm": None,
                },
                "generation": {
                    "batch_size": 4,
                    "gen_kwargs": {"max_new_tokens": 512, "top_k": 0, "top_p": 1.0, "do_sample": False},
                    "gen_experience_kwargs": {
                        "max_new_tokens": 512,
                        "do_sample": False,
                        "temperature": 1.0,
                        "top_k": 50,
                        "top_p": 0.95,
                    },
                },
                "model_keys": ["actor", "critic", "ref_model", "reward_model"],
                "tokenizer": {"params": {"truncation_side": "right"}, "tokenizer_path": "/home/glm-large-chinese"},
            },
        )
        self.assertEqual(atorch_rl_config.generation.batch_size, 4)

    def test_share_wights_model_config(self):
        config_path = "./atorch/tests/test_define_rl_models/share_weights_models/config.yaml"
        atorch_rl_config = AtorchRLConfig.load_yaml(config_path)
        self.assertTrue(isinstance(atorch_rl_config.model.actor_critic_ref, TrainableModelConfig))
        self.assertEqual(atorch_rl_config.model.actor_critic_ref.loss, "atorch.rl.ppo_utils.ppo_util.loss")
        import_module(atorch_rl_config.model.actor_critic_ref.loss)
        self.assertEqual(atorch_rl_config.model.actor_critic_ref.model_path, "/home/glm-large-chinese")
        self.assertEqual(atorch_rl_config.model.actor_critic_ref.optimizer.name, "torch.optim.adam")
        self.assertEqual(
            atorch_rl_config.model.actor_critic_ref.model_cls,
            "atorch.tests.test_define_rl_models.independent_models.actor_critic_ref.ActorCriticRef",
        )
        self.assertListEqual(atorch_rl_config.model_keys, ["actor_critic_ref", "reward_model"])


if __name__ == "__main__":
    unittest.main()
