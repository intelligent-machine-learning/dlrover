import torch
from torch.nn import LayerNorm

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.optimization import Optimization
from atorch.distributed.distributed import local_rank
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device
from atorch.modules.normalization import LayerNorm as ApexLayerNorm
from atorch.modules.transformer.inject import replace_module
from atorch.modules.transformer.layers import (
    BertAttentionFA,
    CLIPAttentionFA,
    GPT2AttentionFA,
    LlamaAttentionFA,
    MultiheadAttentionFA,
)
from atorch.utils.meta_model_utils import empty_param, recursive_empty_param
from atorch.utils.version import package_version_smaller_than

# supported replacement pairs, default replace_configs and supported dtypes for module replace optimization
# in format of {pair_name: (src_module_cls, target_cls, kwargs, supported_dtypes)}
REPLACEMENT_PAIRS = dict()


def register_replace_pair(
    pair_name,
    kwargs={"need_src_module": True},
    supported_dtypes={torch.float32, torch.float16, torch.bfloat16},
    pair_cls=None,
):
    """
    kwargs: used as `**kwargs` when calling `replace_module`.

    This func can be used in decorator mode. Decorator mode requires that pair_cls is None,
    and the target cls must inherit from source cls or has `_src_module_cls` attribute:

        >>> @register_replace_pair('pair_name')
        >>> class TgtCls(SrcCls):
        >>>    pass
        >>> # or
        >>> @register_replace_pair('pair_name')
        >>> class TgtCls(torch.nn.Module):
        >>>    _src_module_cls = SrcCls
        >>>    pass

    And this func can directly register the replace pair when assigning pair_cls:

        >>> register_replace_pair('pair_name', pair_cls=(SrcCls, TgtCls))
    """
    global REPLACEMENT_PAIRS

    if pair_cls is None:
        # decorator mode
        def decorator(cls):
            src_cls = getattr(cls, "_src_module_cls", cls.__base__)
            REPLACEMENT_PAIRS[pair_name] = (src_cls, cls, kwargs, supported_dtypes)
            return cls

        return decorator
    else:
        src_cls, cls = pair_cls
        REPLACEMENT_PAIRS[pair_name] = (src_cls, cls, kwargs, supported_dtypes)


# directly register
register_replace_pair("LayerNorm_Apex", kwargs={"init_from_attr": True}, pair_cls=(LayerNorm, ApexLayerNorm))

# decorator mode register. Not doing this in cls definition
# because importing `register_replace_pair` there incurs circular import
register_replace_pair("HF_BertAttention_FA", supported_dtypes={torch.float16, torch.bfloat16})(BertAttentionFA)
register_replace_pair("HF_CLIPAttention_FA", supported_dtypes={torch.float16, torch.bfloat16})(CLIPAttentionFA)
register_replace_pair("MultiheadAttention_FA", supported_dtypes={torch.float16, torch.bfloat16})(MultiheadAttentionFA)
if package_version_smaller_than("transformers", "4.38.0"):
    # transformers 4.38.0 changed LlamaAttention interface, so check version first.
    register_replace_pair("HF_LlamaAttention_FA", supported_dtypes={torch.float16, torch.bfloat16})(LlamaAttentionFA)
register_replace_pair("HF_GPT2Attention_FA", supported_dtypes={torch.float16, torch.bfloat16})(GPT2AttentionFA)


def _check_model_params_device(model):
    devices = set()

    for param in model.parameters():
        devices.add(str(param.device))
    return devices


def _enable_flash_attn_by_attr(model):
    model.apply(lambda m: setattr(m, "use_fa", True) if hasattr(m, "use_fa") else None)


def _replace_by_config(model, config=None, gpu_used=False, verbose=True):
    global REPLACEMENT_PAIRS
    cur_dtype = (
        AutoAccelerateContext.half_precision_dtype[AutoAccelerateContext.counter]
        if hasattr(AutoAccelerateContext, "half_precision_dtype")
        and AutoAccelerateContext.counter in AutoAccelerateContext.half_precision_dtype
        else torch.float32
    )
    if config is None:
        config = {pair_name: None for pair_name in REPLACEMENT_PAIRS}

    # pop unsupported dtype pairs.
    for pair in list(config.keys()):
        if cur_dtype not in REPLACEMENT_PAIRS[pair][3]:
            config.pop(pair)
    pre_replacement_devices = _check_model_params_device(model)
    for pair_name in config:
        src_module_cls, tgt_module_cls, default_kwargs, _ = REPLACEMENT_PAIRS[pair_name]
        kwargs = default_kwargs if config[pair_name] is None else config[pair_name]
        model = replace_module(model, src_module_cls, tgt_module_cls, verbose=verbose, **kwargs)
    if getattr(AutoAccelerateContext, "FSDP_META_INIT", None) == "NEW_META":
        return model
    # Handles device placement mismatch
    # In case modules are on meta, do nothing and assume defer init is enabled
    if (
        len(pre_replacement_devices) == 1
        and "meta" not in pre_replacement_devices
        and torch.device("meta") not in pre_replacement_devices
    ):
        materialize_modules_to_device(
            model,
            list(pre_replacement_devices)[0],
        )

    post_replacement_devices = _check_model_params_device(model)
    if len(post_replacement_devices) > 1:
        # In this case we assume defer init happens
        if "meta" in pre_replacement_devices or torch.device("meta") in pre_replacement_devices:
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
        verbose = True if wrapper_config is None else wrapper_config.pop("verbose", True)
        if wrapper_config is None or len(wrapper_config) == 0:
            wrapper_config = _get_default_replace_config()

        tie_weights = {}
        from atorch.utils.meta_model_utils import _find_tied_weights, _retie_weights

        if hasattr(AutoAccelerateContext, "FSDP_META_INIT"):
            tie_weights = _find_tied_weights(model_context.model)
            if getattr(AutoAccelerateContext, "FSDP_META_INIT") != "NEW_META":
                model_context.model = model_context.model.to("meta")

        model_context.model = _replace_by_config(
            model_context.model, config=wrapper_config, gpu_used=model_context.gpu_used, verbose=verbose
        )
        _retie_weights(model_context.model, tie_weights)
        model = model_context.model
        _enable_flash_attn_by_attr(model)
        model_context.model = model
        return model_context
