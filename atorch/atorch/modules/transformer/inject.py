import inspect

from deepspeed import DeepSpeedTransformerConfig

from atorch.common.log_utils import default_logger as logger
from atorch.modules.transformer.layers import MixPrecisionTransformerLayer
from atorch.utils.meta_model_utils import empty_param, is_meta, recursive_empty_param, reload_meta_module


def get_bert_layer_weight_offset(layer):
    """make sure layer have same weight"""
    # length=8 HFBertLayerPolicy
    # (q,k,w,
    # attn_ow
    #  attn_nw
    #  inter_w
    #  output_w
    #  norm_w)
    qw = layer.attention.self.query.weight
    qb = layer.attention.self.query.bias
    kw = layer.attention.self.key.weight
    kb = layer.attention.self.key.bias
    vw = layer.attention.self.value.weight
    vb = layer.attention.self.value.bias
    ow = layer.attention.output.dense.weight
    ob = layer.attention.output.dense.bias
    attn_nw = layer.attention.output.LayerNorm.weight
    attn_nb = layer.attention.output.LayerNorm.bias
    intermediate_ff = layer.intermediate.dense
    output_w = layer.output.dense.weight
    output_norm_w = layer.output.LayerNorm.weight
    output_b = layer.output.dense.bias
    output_norm_b = layer.output.LayerNorm.bias
    return (
        (
            qw,
            kw,
            vw,
            ow,
            attn_nw,
            intermediate_ff.weight,
            output_w,
            output_norm_w,
        ),
        (
            qb,
            kb,
            vb,
            ob,
            attn_nb,
            intermediate_ff.bias,
            output_b,
            output_norm_b,
        ),
    )


def replace_with_deepspeed_transformer(
    layer_obj,
    model,
    config,
    micro_batch_size,
    max_seq_length,
    seed,
    preln=False,
    fp16=True,
):
    """
    Usage:

        new_module = replace_with_deepspeed_transformer(
            BertLayer,
            torch_model,
            config,
            batch_size,
            seq_len,
            self.seed,
            preln=False,
            fp16=use_fp16,
        )
        注：
        amp.initialize O2会修改掉model参数的data，造成replace_with_deepspeed_transformer算子持有的数据指针被破坏
        该函数要放在amp.initialize后面
    """
    for name, child in model.named_children():
        if isinstance(child, layer_obj):
            print("REPLACING BertLayer")

            cuda_config = DeepSpeedTransformerConfig(
                batch_size=micro_batch_size,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                heads=config.num_attention_heads,
                attn_dropout_ratio=config.attention_probs_dropout_prob,
                hidden_dropout_ratio=config.hidden_dropout_prob,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                seed=seed,
                # local rank?
                return_tuple=True,
                fp16=fp16,
                pre_layer_norm=False,
                stochastic_mode=True,  # Enable for high performance
            )
            migrate_weight, migrate_bias = get_bert_layer_weight_offset(child)
            new_module = MixPrecisionTransformerLayer(cuda_config, migrate_weight, migrate_bias, mix_precision=True)

            setattr(model, name, new_module)

        else:
            replace_with_deepspeed_transformer(
                layer_obj,
                child,
                config,
                micro_batch_size,
                max_seq_length,
                seed,
                preln,
                fp16,
            )

    return model


def replace_module(
    model, src_module_cls, tgt_module_cls, config=None, need_src_module=False, init_from_attr=False, bkup_ori=False
):
    r"""replace model's src_module to tgt_module.

    Args:
        model (nn.Module): the model to be scanned through for replacement
        src_module_cls (class): the class of the source module
        tgt_module_cls (class): the class of the target module
        need_src_module (bool, optional):
            Default: ``False``. If ``True``, pass the source module for setting
            state dict or so on. make sure that src_module_cls has 'src_module' kwarg
        init_from_attr (bool, optional):
            Default: ``False``. If ``True``, inspect init args that tgt module needs and look
            up from src module's attrs (e.g. torch LN -> apex LN), and manual load state dict.
        config (CfgObject, optional):
            Default: None. The passed may be accessed for hype-rparameters, mock
            HFace Transformers
        bkup_ori (str, optional): Default: False. Whether to backup original module

    Return:
        model (nn.Module): the replaced model.
        ori_module_bkup (Dict, optional): the backuped original modules, in format of
            {model1: [(child_name, ori_child1), (child_name, ori_child2), ...], model2 ...}

    Note:
        if the model itself is a src_module_cls instance, it wont be replaced.

    Usage:
        >>> import src_module_cls, tgt_module_cls
        >>> model = creat_model()
        >>> new_model = replace_module(model, src_module_cls, tgt_module_cls,
        >>>     config=model.config, need_src_module=True)
        >>> new_model.to(device)  # in case some replacement not handling device
    """
    root_name = model.__class__.__name__
    if bkup_ori:
        ori_module_bkup = dict()
    kwargs = dict()
    if config:
        kwargs["config"] = config

    def _replace(model, cur_name):
        for name, child in model.named_children():
            child_name = cur_name + "." + name
            if isinstance(child, src_module_cls) and not isinstance(child, tgt_module_cls):
                logger.info(f"REPLACING {src_module_cls} '{child_name}' to {tgt_module_cls}")
                if need_src_module:
                    kwargs["src_module"] = child
                if init_from_attr:
                    for arg in inspect.signature(tgt_module_cls).parameters.keys():
                        assert hasattr(
                            child, arg
                        ), f"{tgt_module_cls.__name__}'s arg {arg} is not the attribute of {child.__class__.__name__}"
                        kwargs[arg] = getattr(child, arg)
                new_module = tgt_module_cls(**kwargs)
                if init_from_attr:
                    mod_is_meta = is_meta(child)
                    if mod_is_meta:
                        reload_meta_module(child)
                    new_module.load_state_dict(child.state_dict())
                    if mod_is_meta:
                        recursive_empty_param(child, ignore_save=True)
                        empty_param(new_module, prefix_name="replace_")
                        recursive_empty_param(new_module, prefix_name="replace_")
                if bkup_ori:
                    ori_module_bkup[model] = ori_module_bkup.get(model, []) + [(child_name, child)]
                setattr(model, name, new_module)
            else:
                _replace(child, child_name)

    _replace(model, root_name)
    if bkup_ori:
        return model, ori_module_bkup
    else:
        return model


def reset_replace(ori_module_bkup, to_device=False):
    for model in ori_module_bkup:
        for child_name, child in ori_module_bkup[model]:
            logger.info(f"RESET '{child_name}' back to {child.__class__.__name__}")
            module_name = child_name.split(".")[-1]
            if to_device:
                device = list(getattr(model, module_name).parameters())[0].device
                child.to(device)
            setattr(model, module_name, child)
