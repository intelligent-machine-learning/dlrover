import torch

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.optimization import Optimization
from atorch.distributed.distributed import local_rank
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device
from atorch.modules.transformer.inject import replace_module
from atorch.modules.transformer.layers import BertAttentionFA, CLIPAttentionFA, GPT2AttentionFA, MultiheadAttentionFA
from atorch.normalization import LayerNorm as ApexLayerNorm
from atorch.utils.meta_model_utils import empty_param, recursive_empty_param

try:
    from transformers.modeling_bert import BertAttention  # 3.5
    from transformers.modeling_clip import CLIPAttention
    from transformers.modeling_gpt2 import GPT2Attention
except (ModuleNotFoundError, ImportError):
    from transformers.models.bert.modeling_bert import BertAttention  # 4.17
    from transformers.models.clip.modeling_clip import CLIPAttention
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from torch.nn import LayerNorm, MultiheadAttention

# supported replacement pairs, default replace_configs and supported dtypes for module replace optimization
REPLACEMENT_PAIRS = {
    "HF_BertAttention_FA": (BertAttention, BertAttentionFA, {"need_src_module": True}, {torch.float16, torch.bfloat16}),
    "HF_CLIPAttention_FA": (CLIPAttention, CLIPAttentionFA, {"need_src_module": True}, {torch.float16, torch.bfloat16}),
    "MultiheadAttention_FA": (
        MultiheadAttention,
        MultiheadAttentionFA,
        {"need_src_module": True},
        {torch.float16, torch.bfloat16},
    ),
    "HF_GPT2Attention_FA": (GPT2Attention, GPT2AttentionFA, {"need_src_module": True}, {torch.float16, torch.bfloat16}),
    "LayerNorm_Apex": (LayerNorm, ApexLayerNorm, {"init_from_attr": True}, {torch.float32, torch.float16}),
}


def _check_model_params_device(model):
    devices = set()
    for param in model.parameters():
        devices.add(str(param.device))
    return devices


def _enable_flash_attn_by_attr(model):
    model.apply(lambda m: setattr(m, "use_fa", True) if hasattr(m, "use_fa") else None)


def _replace_by_config(model, config=None, gpu_used=False):
    global REPLACEMENT_PAIRS
    cur_dtype = (
        AutoAccelerateContext.half_precision_dtype[AutoAccelerateContext.counter]
        if hasattr(AutoAccelerateContext, "half_precision_dtype")
        and AutoAccelerateContext.counter in AutoAccelerateContext.half_precision_dtype
        else torch.float32
    )
    if config is None:
        config = {pair_name: None for pair_name in REPLACEMENT_PAIRS}

    # pop unsupported dtype pairs. e.g. apex FusedLayerNorm does not support BFloat16
    for pair in list(config.keys()):
        if cur_dtype not in REPLACEMENT_PAIRS[pair][3]:
            config.pop(pair)
    pre_replacement_devices = _check_model_params_device(model)
    for pair_name in config:
        src_module_cls, tgt_module_cls, default_kwargs, _ = REPLACEMENT_PAIRS[pair_name]
        kwargs = default_kwargs if config[pair_name] is None else config[pair_name]
        model = replace_module(model, src_module_cls, tgt_module_cls, **kwargs)
    # Handles device placement mismatch
    # In case modules are on meta, do nothing and assume defer init is enabled
    if (
        len(pre_replacement_devices) == 1
        and "meta" not in pre_replacement_devices
        and torch.device("meta") not in pre_replacement_devices
    ):
        materialize_modules_to_device(model, list(pre_replacement_devices)[0])

    post_replacement_devices = _check_model_params_device(model)
    if len(post_replacement_devices) > 1:
        # In this case we assume defer init happens
        if "meta" in pre_replacement_devices or torch.device("meta") not in pre_replacement_devices:
            empty_param(model, prefix_name="replace_")
            recursive_empty_param(model, prefix_name="replace_")
        elif torch.cuda.is_available() and not gpu_used:
            materialize_modules_to_device(model, torch.device(type="cuda", index=local_rank()))
        else:
            materialize_modules_to_device(model)
    return model


def _get_default_replace_config():
    return {pair_name: None for pair_name in REPLACEMENT_PAIRS}


class ModuleReplaceOptimization(Optimization):
    def __init__(self):
        super().__init__(name="module_replace", group="module_replace", is_tunable=False)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        model_context.add_wrapper(
            "module_replace", ModuleReplaceOptimization.apply_wrapper, config, is_pre_wrapper=True
        )
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config):
        """
        wrapper_config (Dict[str, Dict], optional): {pair_name: replace_config}
            if None, replace all supported pairs with default replace_config
        """
        if wrapper_config is None:
            wrapper_config = _get_default_replace_config()
        model_context.model = _replace_by_config(
            model_context.model, config=wrapper_config, gpu_used=model_context.gpu_used
        )
        model = model_context.model
        _enable_flash_attn_by_attr(model)
        model_context.model = model
        return model_context
