from atorch.rl.trainer import RLTrainer


class PPOTrainer(RLTrainer):
    def __init__(self, model_engine, dataset, config):
        super().__init__(model_engine, dataset, config)

    def make_experience(self, data):
        pass

    def rl_training(self):
        pass
