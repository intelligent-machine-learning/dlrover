import contextlib
from abc import abstractmethod

import torch.nn
from accelerate.state import GradientState
from torch.utils.data import DataLoader
from transformers.trainer_callback import TrainerState

from atorch.utils.dev_utils import raise_not_impl


class AtorchTrainEngine(torch.nn.Module):
    def __init__(self, train_args):
        super().__init__()
        from atorch.trainer import AtorchTrainingArgs

        self.train_args: AtorchTrainingArgs = train_args
        self.gradient_state = GradientState()
        self.train_state = TrainerState()
        self.pytorch_model = None

    @staticmethod
    def initialize(**kwargs):
        pass

    def optimizer_step(self):
        """
        Subclass should implement the following five step:
        1. unscaling gradient
        2. find inf/nan in gradients (optional)
        3. clip gradient
        4. count zero in gradients (optional)
        5. optimizer.step()
        """
        pass

    def scheduler_step(self):
        pass

    def optimizer_zero_grad(self):
        pass

    def backward(self, loss):
        pass

    @classmethod
    def build_dataloader(cls, dataset, shuffle, collate_fn, batch_size, **kwargs):
        return DataLoader(dataset, shuffle=shuffle, collate_fn=collate_fn, batch_size=batch_size, **kwargs)

    @abstractmethod
    def get_dataloader(self, name=None):
        pass

    @classmethod
    def from_config(cls, atorch_train_step):
        pass

    def forward(self, **batch_data):
        pass

    def train(self):
        pass

    def train_step(self, **batch_data):
        raise NotImplementedError("Subclass should implement this method.")

    def eval(self):
        pass

    def eval_step(self, **batch_data):
        pass

    def save_checkpoint(self, output_dir, trainer_state: dict = None, **kwargs):
        raise NotImplementedError("Subclass should implement this method.")

    def get_checkpoint_path_dir(self, output_dir, **kwargs):
        raise NotImplementedError("Subclass should implement this method.")

    def load_checkpoint(self, input_dir, **kwargs):
        raise NotImplementedError("Subclass should implement this method.")

    @raise_not_impl
    def save(self, output_dir, train_args):
        pass

    @raise_not_impl
    def load(self, input_dir, **load_model_func_kwargs):
        pass

    def training_log(self, **kwargs):
        pass

    @contextlib.contextmanager
    def _no_sync(self, model):
        context = contextlib.nullcontext
        if self.use_distributed:
            context = getattr(model, "no_sync", context)

        with context():
            yield

    def _do_sync(self):
        """
        set the sync_gradients flag
        """
        if self.gradient_state.sync_with_dataloader and self.gradient_state.end_of_dataloader:
            self.train_state.steps_in_epoch = 0
            self.gradient_state._set_sync_gradients(True)
        else:
            # TODO:
            self.step += 1
            self.gradient_state._set_sync_gradients(
                (self.train_state.steps_in_epoch % self.gradient_state.num_steps) == 0
            )

    @contextlib.contextmanager
    def accumulate(self):
        """
        A context manager that will lightly wrap around and perform gradient accumulation automatically

        Args:
            *models (list of `torch.nn.Module`):
            # TODO
                PyTorch Modules that were prepared with `Accelerator.prepare`. Models passed to `accumulate()` will
                skip gradient syncing during backward pass in distributed training
        """
        self._do_sync()

        allow_gradient_sync = self.gradient_state.sync_gradients or (
            # must sync if sync gradients need to complete an optimizer step
            # the no_sync context stops the gradients from reducing during distributed training
            # bringing speedup (potentially at some costs). Here, no_sync can be prevented
            # by setting sync_each_batch = True.
            self.train_args.use_distributed  # only relevant in distributed settings
            and self.gradient_state.plugin_kwargs.get("sync_each_batch", False)
        )
        with contextlib.ExitStack() as cm_stack:
            cm_stack.enter_context(
                contextlib.nullcontext() if allow_gradient_sync else self._no_sync(self.pytorch_model)
            )
            yield
