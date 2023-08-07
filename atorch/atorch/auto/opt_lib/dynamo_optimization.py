# integrate the dynamo optimization into atorch optimization library
import torch

from atorch.auto.opt_lib.optimization import Optimization
from atorch.common.log_utils import default_logger as logger

from .dynamo_backends import DynamoBackends


class NativeDynamoOptimization(Optimization):
    """Support native dynamo optimization

    Args:
        dynamo_backend (str): By default use inductor as the default compiler
    """

    def __init__(self, dynamo_backend="inductor"):
        group = "dynamo"
        name = "native_dynamo"
        is_tunable = True
        self.dynamo_backend = DynamoBackends(dynamo_backend.upper()) if dynamo_backend else dynamo_backend
        super().__init__(name, group, is_tunable)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        for opt in strategy:
            opt_name = opt[0]
            if opt_name == "amp_native":
                self.mixed_precision = True
            else:
                self.mixed_precision = False
        config = dict() if config is None else config
        config["mixed_precision"] = self.mixed_precision
        config["dynamo_backend"] = self.dynamo_backend
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=dict()):
        model_context.add_wrapper("native_dynamo", NativeDynamoOptimization.apply_wrapper, config, is_pre_wrapper=True)
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config):
        try:
            # setup tf32
            cuda_available = torch.cuda.is_available()
            mixed_precision = wrapper_config.get("mixed_precision", False)
            dynamo_backend = wrapper_config.get("dynamo_backend", None)
            allow_tf32 = wrapper_config.get("allow_tf32", False)
            if (
                dynamo_backend is not None
                and dynamo_backend != DynamoBackends.NO
                and not mixed_precision
                and cuda_available
                and allow_tf32
            ):
                # allow tf32
                torch.backends.cuda.matmul.allow_tf32 = True
            if dynamo_backend is not None and dynamo_backend != DynamoBackends.NO:
                import torch._dynamo as dynamo

                dynamo.reset()
                model_context.model = dynamo.optimize(dynamo_backend.value.lower())(model_context.model)
        except ImportError:
            logger.warning("No dynamo tool, raise torch version to 2.0 or above")
