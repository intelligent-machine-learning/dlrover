import copy
from dataclasses import dataclass
from typing import Any

import yaml  # type: ignore[import]

from atorch.rl.model_utils.model_util import is_trainable_model


def recursive_transform_to_dict(value):
    new_value = value
    if hasattr(value, "to_dict"):
        value = value.to_dict()

    if isinstance(value, dict):
        new_value = copy.deepcopy(value)
        for k, v in value.items():
            new_value[k] = recursive_transform_to_dict(v)
    return new_value


class Base:
    def to_dict(self):
        new_attr = {}
        attr = self.__dict__
        for k, v in attr.items():
            v = recursive_transform_to_dict(v)
            if not k.startswith("__") and not isinstance(v, classmethod):
                new_attr[k] = v
        return new_attr


@dataclass
class Optimizer(Base):
    name: str
    kwargs: dict

    @classmethod
    def from_dict(cls, config):
        return cls(**config)


@dataclass
class GeneratationConfig(Base):
    batch_size: int
    gen_kwargs: dict
    gen_experience_kwargs: dict

    @classmethod
    def update_config_with_default_value(cls, config):

        default_value = {
            "batch_size": 4,
            "gen_kwargs": {"max_new_tokens": 512, "top_k": 0, "top_p": 1.0, "do_sample": False},
            "gen_experience_kwargs": {
                "max_new_tokens": 512,
                "do_sample": False,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 0.95,
            },
        }
        for k, v in default_value.items():
            if config.get(k) is None:
                config[k] = v

        return config

    @classmethod
    def from_dict(cls, config):
        config = cls.update_config_with_default_value(config)
        return cls(
            batch_size=config["batch_size"],
            gen_kwargs=config["gen_kwargs"],
            gen_experience_kwargs=config["gen_experience_kwargs"],
        )


@dataclass
class TrainConfig(Base):
    seq_length: int
    batch_size: int
    epoch: int
    num_rollouts: int
    mode: str
    trainer: str
    logdir: str
    scheduler: dict
    eval_interval: int
    checkpoint_interval: int
    checkpoint_dir: str
    gradient_accumulation_steps: int
    max_grad_norm: Any

    @classmethod
    def update_config_with_default_value(cls, config):
        default_value = {
            "trainer": "PPOTrainer",
            "logdir": "./tensorboard",
            "mode": "Concurrent",
            "seq_length": 1024,
            "num_rollouts": 2048,
            "batch_size": 1024,
            "epoch": 100,
            "scheduler": {
                "name": "cosine_warmup",
                "kwargs": {"num_warmup_steps": 640, "num_training_steps": 6400},
            },
            "eval_interval": 100,
            "checkpoint_interval": 100,
            "checkpoint_dir": "./checkpoint",
            "gradient_accumulation_steps": 1,
            "max_grad_norm": None,
        }

        for k, v in default_value.items():
            if config.get(k) is None:
                config[k] = v
        return config

    @classmethod
    def from_dict(cls, config):
        config = cls.update_config_with_default_value(config)
        return cls(**config)


@dataclass
class TokenizerConfig(Base):
    params: dict
    tokenizer_path: str

    @classmethod
    def from_dict(cls, config):
        return cls(**config)


@dataclass
class ModelConfig(Base):
    model_path: str
    model_cls: str
    model_params: dict
    train_strategy: str
    inference_strategy: str
    lazy_load: bool
    peft_config: Any

    @classmethod
    def update_config_with_default_value(cls, config):
        default_value = {
            "model_params": {},
            "train_strategy": "torch_native",
            "lazy_load": False,
            "inference_strategy": "torch_native",
            "peft_config": None,
        }
        for k, v in default_value.items():
            if config.get(k) is None:
                config[k] = v
        return config

    @classmethod
    def from_dict(cls, config):
        config = cls.update_config_with_default_value(config)
        return cls(**config)


@dataclass
class TrainableModelConfig(Base):
    optimizer: Optimizer
    model_path: str
    model_cls: str
    model_params: dict
    peft_config: Any
    train_strategy: str
    inference_strategy: str
    lazy_load: bool
    loss: str

    @classmethod
    def update_config_with_default_value(cls, config):
        default_value = {
            "optimizer": {"name": "torch.optim.adam", "kwargs": {"betas": (0.9, 0.999)}},
            "train_strategy": "torch_native",
            "inference_strategy": "torch_native",
            "lazy_load": False,
            "model_params": {},
            "loss": "",
            "peft_config": None,
        }
        for k, v in default_value.items():
            if config.get(k) is None:
                config[k] = v
        return config

    @classmethod
    def from_dict(cls, config):
        config = cls.update_config_with_default_value(config)
        optimizer = config["optimizer"]
        optimizer = Optimizer.from_dict(optimizer)
        config["optimizer"] = optimizer
        return cls(**config)


@dataclass
class Model(Base):
    model: dict

    def __getattr__(cls, key):
        return cls.model[key]


@dataclass
class PPOConfig(Base):
    ppo_epoch: int
    init_kl_coef: float
    gamma: float
    lam: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    cliprange_reward: float
    ent_coef: float
    horizon: float
    clip_ratio: bool
    scale_reward: str
    ref_mean: Any
    ref_std: Any

    @classmethod
    def update_config_with_default_value(self, config):
        default_value = {"scale_reward": "running", "horizon": 10000.0}
        for k, v in default_value.items():
            if config.get(k) is None:
                config[k] = v

    @classmethod
    def from_dict(cls, config):
        cls.update_config_with_default_value(config["PPOConfig"])
        return cls(**config["PPOConfig"])


@dataclass
class AtorchRLConfig(Base):
    """
    Top level config for atorch rlhf. Loads configs and can be converted to dictionary.
    """

    model: dict
    ppo_config: PPOConfig
    train: TrainConfig
    generation: GeneratationConfig
    model_keys: list
    tokenizer: TokenizerConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        """
        Load yaml file as ArorchRLConfig.
        :param yml_fp: Path to yaml file
        :type yml_fp: str
        """
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

    @classmethod
    def update_config_with_default_value(cls, config):
        return config

    @classmethod
    def from_dict(cls, config):
        """
        Convert dictionary to TRLConfig.
        """
        cls.update_config_with_default_value(config)
        model_keys = [i for i in config["model"].keys()]
        model = {}
        for m in model_keys:
            if is_trainable_model(m):
                model.update({m: TrainableModelConfig.from_dict(config["model"].get(m))})
            else:
                model.update({m: ModelConfig.from_dict(config["model"].get(m))})
        return cls(
            model=Model(model=model),
            ppo_config=PPOConfig.from_dict(config["method"]),
            tokenizer=TokenizerConfig.from_dict(config["tokenizer"]),
            train=TrainConfig.from_dict(config["train"]),
            generation=GeneratationConfig.from_dict(config["generation"]),
            model_keys=model_keys,
        )
