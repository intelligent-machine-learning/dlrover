from abc import ABC, abstractmethod


class Optimization(ABC):
    def __init__(self, name, group, is_tunable, is_distributed=False):
        self.name = name
        self.group = group
        self.is_tunable = is_tunable
        self.is_distributed = is_distributed

    def distributed_only(self, config=None):
        return self.is_distributed

    @abstractmethod
    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        """
        Find best config for this optimization for model_context.
        config: an imcomplete config or None.
        strategy: a list of optimization methods, and self.name is in it.
                  all methods before self.name are already applied on model_context
                  all methods after self.name will be applied after
        apply_transform: if apply transform to model_context with apply_wrapper=False
        time_limit: if not None, should finish tune within this seconds
        Return: status, best_config, model_context
            status: if tune successfully.
            best_config: the successfully tuned config for this optimization
            model_context: if apply_transform, transformed model_context; original model_context otherwise.
        """
        pass

    @abstractmethod
    def transform(self, model_context, config=None):
        """
        Return the transformed model_context.
        Note that is some transformation must applied after all other optimization methods
        are done, or the transformation must use instanced model/optim/dataloader, a wrapper
        can be added to model_context, so the transformation will be applied later.
        """
        pass

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        """
        Apply a wrapper (whose name is wrapper_name) with wrapper_config to model_context
        and return the wrappered model_context
        """
        return model_context

    @staticmethod
    def reset(config):
        """Reset environment changed by current optimization."""
        pass
