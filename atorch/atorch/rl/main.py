import atorch
from atorch.rl.config import AtorchRLConfig
from atorch.rl.data import create_dataset
from atorch.rl.model_engine import ModelEngine
from atorch.rl.trainer.ppo_trainer import PPOTrainer

config_file = "my_config.yml"


config = AtorchRLConfig.load_yaml(config_file)

atorch.init_distributed()

# create model, optimizer, tokenizer, etc.
engine = ModelEngine(config)

# create prompt dataset
dataset = create_dataset(config)

# init trainer
trainer = PPOTrainer(engine, dataset, config)

trainer.train()
