from abc import ABC, abstractmethod


class AtorchTrainStep(ABC):
    """
    Abstract base class of train step to regulate the three main function: get batch, loss func, and forward step.
    Users can customize their TrainStep by inheriting AtorchTrainStep and implementing the abstract functions.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_batch_func(self, **kwargs):
        pass

    @abstractmethod
    def get_loss_func(self, **kwargs):
        pass

    @abstractmethod
    def get_forward_step_func(self, **kwargs):
        pass
