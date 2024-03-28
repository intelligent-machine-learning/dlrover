import argparse

import atorch
from atorch.rl.config import AtorchRLConfig
from atorch.rl.data import create_dataset
from atorch.rl.model_engine import ModelEngine
from atorch.rl.trainer.ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("--config_file", type=str, default="my_config.yml", required=False)
    return parser.parse_args()


def rl_train(args):
    config = AtorchRLConfig.load_yaml(args.config_file)

    atorch.init_distributed()

    # create model, optimizer, tokenizer, etc.
    engine = ModelEngine(config)

    # create prompt dataset
    dataset = create_dataset(config)

    # init trainer
    trainer = PPOTrainer(engine, dataset, config)

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    rl_train(args)
