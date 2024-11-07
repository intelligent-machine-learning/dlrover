import torch.optim
from torch.optim import Optimizer

from atorch.trainer.base.inferface import Stateful


class AtorchOptimizer(Optimizer, Stateful):
    def __init__(self, optimizer: Optimizer, scaler):
        self.optimizer = optimizer
        self.scaler = scaler

    @classmethod
    def from_config(cls, distributed_type, configs=None, *args, **kwargs):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        if configs is None:
            model_named_parameters = kwargs.get("model_named_parameters")
            no_decay = ["bias", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model_named_parameters if not any(nd in n for nd in no_decay)],
                    "weight_decay": configs.weight_decay,
                },
                {
                    "params": [p for n, p in model_named_parameters if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=kwargs.get("learning_rate"))
            return optimizer
        else:
            optimizer = torch.optim.AdamW(configs)

        return optimizer

    def state_dict(self, *args, **kwargs):
        raise NotImplementedError

    def load_state_dict(self, *args, **kwargs):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True):
        pass

    def step(self, closure: None = ...) -> None:  # type: ignore[override]
        pass

    def train(self):
        self.optimizer.train()

    def eval(self):
        self.optimizer.eval()
