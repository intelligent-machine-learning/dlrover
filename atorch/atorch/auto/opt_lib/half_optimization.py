from functools import partial

import torch

from atorch.auto.opt_lib.optimization import Optimization
from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import data_float_to_dtype


class HalfOptimization(Optimization):
    """HalfOptimization will convert model to half (fp16 or bf16)
    config is a string, "fp16" (default) or "bf16".
    """

    checkpoint_funcs_before_overrided = None

    def __init__(self):
        super().__init__("half", "half", False)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config="fp16"):
        model_context.add_wrapper("half", HalfOptimization.apply_wrapper, wrapper_config=config, is_pre_wrapper=True)
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        # wrapper_config should be one of "fp16", "bf16".
        if wrapper_config not in ("fp16", "bf16"):
            logger.error("Invalid config for half optimization. Should be fp16 or bf16 but get %s", wrapper_config)
        dtype = torch.float16 if wrapper_config == "fp16" else torch.bfloat16
        from atorch.utils.meta_model_utils import custom_transform_model_keep_checkpoint_name

        model_context.model = custom_transform_model_keep_checkpoint_name(model_context.model, lambda m: m.to(dtype))

        def inputs_to_dtype(data, device, dtype, ori_func):
            data = ori_func(data, device)
            return data_float_to_dtype(data, dtype)

        model_context.prepare_input = partial(inputs_to_dtype, dtype=dtype, ori_func=model_context.prepare_input)

        return model_context
