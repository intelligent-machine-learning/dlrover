import atorch
from atorch.common.log_utils import DashBoardWriter
from atorch.rl.model_engine import ModelEngineState
from atorch.rl.replay_buffer import ReplayBuffer


class RLTrainer:
    def __init__(self, model_engine, dataset, config):
        """
        dataset: dict, keys are train and eval
        """
        self.model_engine = model_engine
        self.dataset = dataset
        self.train_dataset_length = len(self.dataset["train"])
        self.evaluation_dataset_length = len(self.dataset["eval"])
        self.config = config

        self.train_epoch = config.train.epoch
        self.num_rollouts = config.train.num_rollouts
        self.n_updates_per_batch = config.ppo_config.ppo_epoch

        self.replay_buffer = self.create_replay_buffer()
        self.rl_dataloader = None
        self.rl_state = False
        self.dashboard_writer = DashBoardWriter(logdir=config.train.logdir)
        self.initial_start = True

    def get_train_dataset(self):
        return self.dataset["train"]

    def get_eval_dataset(self):
        return self.dataset["eval"]

    def create_replay_buffer(self):
        return ReplayBuffer(self.config)

    def make_experience(self, data):
        pass

    def rl_training(self):
        pass

    def pre_make_expereince_hook(self):
        if self.rl_state:
            self.rl_state = False
            self.model_engine.pre_make_experience_hook()

    def post_experience_generation_hook(self):
        self.model_engine.post_make_experience_hook()

    def pre_rl_training_hook(self):
        self.rl_state = True
        self.model_engine.pre_rl_training_hook()
        self.replay_buffer.sync()
        rl_dataset = self.replay_buffer.create_dataset()
        self.rl_dataloader = self.model_engine.create_dataloader(rl_dataset, state=ModelEngineState.RLTrainingState)

    def post_rl_training_hook(self):
        self.rl_dataloader = None
        self.replay_buffer.reset()
        self.model_engine.post_rl_training_hook()

    def evaluate(self):
        # Todo: add evaulation prompts and dataloader
        # Use table to show results in logs or console

        pass

    def train(self):
        self.model_engine.set_state(ModelEngineState.ExperienceGenerationState)
        dataloader = self.model_engine.create_dataloader(
            self.get_train_dataset(), state=ModelEngineState.ExperienceGenerationState
        )

        from tqdm import tqdm

        for _ in range(self.train_epoch):
            pbar = tqdm(total=self.num_rollouts, desc="RL_Making_Experience", mininterval=0.01)
            for data in dataloader:
                self.pre_make_expereince_hook()
                rank = atorch.local_rank()
                data.to(rank)
                self.make_experience(data)
                pbar.update(self.config.generation.batch_size)
                if self.replay_buffer.num >= self.num_rollouts:
                    pbar = tqdm(total=self.num_rollouts, desc="RL_Making_Experience", mininterval=0.01)
                    self.post_experience_generation_hook()
                    self.pre_rl_training_hook()
                    self.rl_training()
                    self.post_rl_training_hook()
