# coding=utf-8
from __future__ import absolute_import, unicode_literals

import copy
import inspect
from importlib.metadata import version

import torch
import torch.nn.functional as F
from packaging.version import parse as parse_version
from torch import nn
from torch.nn import MultiheadAttention

from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import divide, split_tensor_along_last_dim
from atorch.utils.fa_util import patch_fa_interface_to_autocast
from atorch.utils.import_util import is_torch_npu_available, is_torch_xla_available, is_xla_device_available
from atorch.utils.meta_model_utils import is_meta, recursive_empty_param, reload_meta_module

try:
    from deepspeed import DeepSpeedTransformerLayer
    from deepspeed.ops.op_builder import StochasticTransformerBuilder, TransformerBuilder
    from deepspeed.ops.transformer import transformer  # using module var
except (ImportError, ModuleNotFoundError):
    DeepSpeedTransformerLayer = StochasticTransformerBuilder = TransformerBuilder = object
    transformer = None


class ImportFailDummyClass:
    pass


try:
    from transformers.modeling_bert import BertAttention  # 3.5
    from transformers.modeling_clip import CLIPAttention
    from transformers.modeling_gpt2 import GPT2Attention
except (ModuleNotFoundError, ImportError):
    from transformers.models.bert.modeling_bert import BertAttention  # 4.17
    from transformers.models.clip.modeling_clip import CLIPAttention
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

try:
    from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
except (ModuleNotFoundError, ImportError):
    LlamaAttention = ImportFailDummyClass
    logger.info("Transformers is not in llama supported version.")

try:
    import flash_attn
except ModuleNotFoundError:
    logger.info("flash_attn not installed")
    flash_attn = None


if flash_attn is not None:
    # patching flash_attn.flash_attn_interface.flash_attn[xxxx]_func to handle autocast
    import flash_attn.flash_attn_interface

    _flash_attn_version = parse_version(version("flash-attn"))
    try:
        from flash_attn.flash_attention import FlashMHA  # cuda version
    except (ImportError, ModuleNotFoundError):
        assert _flash_attn_version >= parse_version(
            "2"
        ), "FlashMHA is deleted in 2.0 release, but FA1 should has FlashMHA."
        from atorch.modules.transformer._fa_api_compat_patch import FlashMHA

    from flash_attn.bert_padding import pad_input, unpad_input

    if _flash_attn_version >= parse_version("2"):
        # Patch fa interface in kernels
        from atorch.kernels import flash_attn_func
        from atorch.kernels import flash_attn_varlen_func as flash_attn_unpadded_func
    else:
        patch_fa_interface_to_autocast(flash_attn.flash_attn_interface)
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func  # type: ignore

    from atorch.kernels import dropout_add_layer_norm
else:
    FlashMHA = ImportFailDummyClass
    _flash_attn_version = None
    dropout_add_layer_norm = None

try:
    # Patching flash_attn_1.flash_attn_interface.flash_attn[xxxx]_func to handle autocast
    # Patch flash_attn_1 interface func in kernels
    from atorch.kernels import dropout_add_layer_norm_1 as dropout_add_layer_norm  # type: ignore
    from atorch.kernels import flash_attn_unpadded_func_1 as flash_attn_1_unpadded_func

    has_legacy_fa1 = True
    if _flash_attn_version is None:
        _flash_attn_version = parse_version(version("flash-attn-1"))
except (ImportError, ModuleNotFoundError):
    flash_attn_1 = None
    has_legacy_fa1 = False

try:
    from apex.amp import _amp_state
except (ImportError, ModuleNotFoundError):
    _amp_state = None

try:
    import torch_xla.core.xla_model as xm
except (ImportError, ModuleNotFoundError):
    xm = None

if is_torch_xla_available():
    from atorch.kernels import xla_flash_attn, xla_flash_attn_varlen

if is_torch_npu_available():
    from atorch.npu.layers import npu_fa_with_glm_mask


def is_apex_amp_activate():
    if _amp_state is None:
        return False
    return hasattr(_amp_state, "opt_properties") and _amp_state.opt_properties.enabled


def is_additive_mask_bias_supported_fa1():
    if is_torch_npu_available():
        return True
    if not _flash_attn_version or _flash_attn_version >= parse_version("2"):
        return False
    if has_legacy_fa1:
        flash_attn_unpadded_func = flash_attn_1_unpadded_func
    return "attn_mask" in inspect.signature(flash_attn_unpadded_func).parameters


def is_glm_mask_supported_fa2():
    if is_torch_npu_available():
        return True
    if not _flash_attn_version or _flash_attn_version < parse_version("2"):
        return False
    return "glm_mask" in inspect.signature(flash_attn_func).parameters


def is_pack_glm_mask_supported_fa2():
    if not _flash_attn_version or _flash_attn_version < parse_version("2"):
        return False
    return _flash_attn_version.local == "pack.glm.mask"


class MixPrecisionDeepSpeedTransformerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        input_mask,
        self,
        grads,
        layer_id,
        attn_qkvw,
        attn_qkvb,
        attn_ow,
        attn_ob,
        attn_nw,
        attn_nb,
        inter_w,
        inter_b,
        output_w,
        output_b,
        norm_w,
        norm_b,
        config,
    ):

        cuda_module = (
            transformer.stochastic_transformer_cuda_module
            if config.stochastic_mode
            else transformer.transformer_cuda_module
        )
        forward_func = cuda_module.forward_fp16 if config.fp16 else cuda_module.forward_fp32

        inp_size = input.size()
        if inp_size[1] % 16 != 0:
            input = torch.cat(
                (
                    input,
                    torch.randn(
                        (inp_size[0], (16 - (inp_size[1] % 16)), inp_size[2]),
                        device=input.device,
                        dtype=input.dtype,
                    ),
                ),
                1,
            )
            input_mask = torch.cat(
                (
                    input_mask,
                    torch.ones(
                        (
                            inp_size[0],
                            input_mask.shape[1],
                            input_mask.shape[2],
                            (16 - (inp_size[1] % 16)),
                        ),
                        device=input_mask.device,
                        dtype=input_mask.dtype,
                    )
                    * -10000,
                ),
                3,
            )

        (
            output,
            inp_norm,
            qkv_tf,
            soft_inp,
            ctx_bufB,
            attn_o_inp,
            add_res,
            ff1_inp,
            gelu_inp,
            ff2_inp,
            attn_prob_dropout_mask,
            attn_output_dropout_mask,
            layer_output_dropout_mask,
            attn_layer_norm_var,
            attn_layer_norm_mean,
            layer_norm_var,
            layer_norm_mean,
        ) = forward_func(
            config.layer_id,
            input,
            input_mask,
            attn_qkvw,
            attn_qkvb,
            attn_ow,
            attn_ob,
            attn_nw,
            attn_nb,
            inter_w,
            inter_b,
            output_w,
            output_b,
            norm_w,
            norm_b,
            config.training,
            config.pre_layer_norm,
            config.attn_dropout_checkpoint,
            config.normalize_invertible,
            config.gelu_checkpoint,
        )

        # For testing only.
        if grads is not None:
            for i in [2]:
                attn_qkvw.register_hook(
                    lambda x, i=i, self=self: grads.append(
                        [
                            x[
                                slice(
                                    i * attn_ow.size(0),
                                    (i + 1) * attn_ow.size(0),
                                )
                            ],
                            ("Q_W" if i == 0 else "K_W" if i == 1 else "V_W"),
                        ]
                    )
                )
            for i in [2]:
                attn_qkvb.register_hook(
                    lambda x, i=i, self=self: grads.append(
                        [
                            x[
                                slice(
                                    i * attn_ow.size(0),
                                    (i + 1) * attn_ow.size(0),
                                )
                            ],
                            ("Q_B" if i == 0 else "K_B" if i == 1 else "V_B"),
                        ]
                    )
                )

            attn_ow.register_hook(lambda x, self=self: grads.append([x, "O_W"]))
            attn_ob.register_hook(lambda x, self=self: grads.append([x, "O_B"]))
            attn_nw.register_hook(lambda x, self=self: grads.append([x, "N2_W"]))
            attn_nb.register_hook(lambda x, self=self: grads.append([x, "N2_B"]))
            inter_w.register_hook(lambda x, self=self: grads.append([x, "int_W"]))
            inter_b.register_hook(lambda x, self=self: grads.append([x, "int_B"]))
            output_w.register_hook(lambda x, self=self: grads.append([x, "out_W"]))
            output_b.register_hook(lambda x, self=self: grads.append([x, "out_B"]))
            norm_w.register_hook(lambda x, self=self: grads.append([x, "norm_W"]))
            norm_b.register_hook(lambda x, self=self: grads.append([x, "norm_B"]))

        if config.is_grad_enabled and config.training:
            if config.pre_layer_norm and config.normalize_invertible:
                ctx.save_for_backward(
                    input_mask,
                    attn_qkvw,
                    attn_qkvb,
                    attn_ow,
                    attn_ob,
                    attn_nw,
                    attn_nb,
                    inter_w,
                    inter_b,
                    output_w,
                    output_b,
                    norm_w,
                    norm_b,
                )
            else:
                ctx.save_for_backward(
                    output,
                    input,
                    input_mask,
                    attn_qkvw,
                    attn_qkvb,
                    attn_ow,
                    attn_ob,
                    attn_nw,
                    attn_nb,
                    inter_w,
                    inter_b,
                    output_w,
                    output_b,
                    norm_w,
                    norm_b,
                )

            ctx.config = config
            if config.pre_layer_norm or not config.normalize_invertible:
                ctx.inp_norm = inp_norm

            ctx.qkv_tf = qkv_tf
            ctx.soft_inp = soft_inp
            if not config.attn_dropout_checkpoint:
                ctx.ctx_bufB = ctx_bufB

            ctx.attn_o_inp = attn_o_inp
            if not config.normalize_invertible:
                ctx.add_res = add_res

            ctx.attn_layer_norm_mean = attn_layer_norm_mean
            ctx.layer_norm_mean = layer_norm_mean

            ctx.ff1_inp = ff1_inp
            if not config.gelu_checkpoint:
                ctx.gelu_inp = gelu_inp

            ctx.ff2_inp = ff2_inp
            ctx.attn_prob_dropout_mask = attn_prob_dropout_mask
            ctx.attn_output_dropout_mask = attn_output_dropout_mask
            ctx.layer_output_dropout_mask = layer_output_dropout_mask
            ctx.attn_layer_norm_var = attn_layer_norm_var
            ctx.layer_norm_var = layer_norm_var

        if inp_size[1] % 16 != 0:
            output = torch.narrow(output, 1, 0, inp_size[1])

        if config.return_tuple:
            return (output,)  # outputs -> (output) : outputs[0] = output
        else:
            return output

    @staticmethod
    def backward(ctx, grad_output):
        bsz = grad_output.shape[0]
        grad_output_shape = grad_output.size()
        if grad_output_shape[1] % 16 != 0:
            grad_output = torch.cat(
                (
                    grad_output,
                    torch.zeros(
                        (
                            bsz,
                            (16 - (grad_output_shape[1] % 16)),
                            grad_output_shape[2],
                        ),
                        device=grad_output.device,
                        dtype=grad_output.dtype,
                    ),
                ),
                1,
            )

        assert ctx.config.training

        if ctx.config.pre_layer_norm and ctx.config.normalize_invertible:
            (
                input_mask,
                attn_qkvw,
                attn_qkvb,
                attn_ow,
                attn_ob,
                attn_nw,
                attn_nb,
                inter_w,
                inter_b,
                output_w,
                output_b,
                norm_w,
                norm_b,
            ) = ctx.saved_tensors
        else:
            (
                output,
                input,
                input_mask,
                attn_qkvw,
                attn_qkvb,
                attn_ow,
                attn_ob,
                attn_nw,
                attn_nb,
                inter_w,
                inter_b,
                output_w,
                output_b,
                norm_w,
                norm_b,
            ) = ctx.saved_tensors

        cuda_module = (
            transformer.stochastic_transformer_cuda_module
            if ctx.config.stochastic_mode
            else transformer.transformer_cuda_module
        )
        backward_func = cuda_module.backward_fp16 if ctx.config.fp16 else cuda_module.backward_fp32

        (
            grad_input,
            grad_attn_qkvw,
            grad_attn_qkvb,
            grad_attn_ow,
            grad_attn_ob,
            grad_attn_nw,
            grad_attn_nb,
            grad_inter_w,
            grad_inter_b,
            grad_output_w,
            grad_output_b,
            grad_norm_w,
            grad_norm_b,
        ) = backward_func(
            ctx.config.layer_id,
            grad_output,
            (ctx.inp_norm if (ctx.config.pre_layer_norm and ctx.config.normalize_invertible) else output),
            (ctx.inp_norm if (ctx.config.pre_layer_norm or not ctx.config.normalize_invertible) else input),
            ctx.qkv_tf,
            ctx.soft_inp,
            (ctx.soft_inp if ctx.config.attn_dropout_checkpoint else ctx.ctx_bufB),
            ctx.attn_o_inp,
            (ctx.ff1_inp if ctx.config.normalize_invertible else ctx.add_res),
            ctx.ff1_inp,
            (ctx.ff2_inp if ctx.config.gelu_checkpoint else ctx.gelu_inp),
            ctx.ff2_inp,
            ctx.attn_prob_dropout_mask,
            ctx.attn_output_dropout_mask,
            ctx.layer_output_dropout_mask,
            ctx.attn_layer_norm_var,
            ctx.attn_layer_norm_mean,
            ctx.layer_norm_var,
            ctx.layer_norm_mean,
            (ctx.inp_norm if (ctx.config.pre_layer_norm and ctx.config.normalize_invertible) else input),
            input_mask,
            attn_qkvw,
            attn_qkvb,
            attn_ow,
            attn_ob,
            attn_nw,
            attn_nb,
            inter_w,
            inter_b,
            output_w,
            output_b,
            norm_w,
            norm_b,
        )

        # This appears to be an effective way to release context memory
        ctx.qkv_tf = None
        ctx.soft_inp = None
        ctx.ctx_bufB = None
        ctx.gelu_inp = None
        ctx.ff2_inp = None
        ctx.attn_o_inp = None
        ctx.ff1_inp = None
        ctx.add_res = None
        ctx.inp_norm = None
        ctx.config = None
        ctx.attn_layer_norm_mean = None
        ctx.layer_norm_mean = None
        ctx.attn_prob_dropout_mask = None
        ctx.attn_output_dropout_mask = None
        ctx.layer_output_dropout_mask = None
        ctx.attn_layer_norm_var = None
        ctx.layer_norm_var = None

        if grad_output_shape[1] % 16 != 0:
            grad_input = torch.narrow(grad_input, 1, 0, grad_output_shape[1])

        return (
            grad_input,
            None,
            None,
            None,
            None,
            grad_attn_qkvw,
            grad_attn_qkvb,
            grad_attn_ow,
            grad_attn_ob,
            grad_attn_nw,
            grad_attn_nb,
            grad_inter_w,
            grad_inter_b,
            grad_output_w,
            grad_output_b,
            grad_norm_w,
            grad_norm_b,
            None,
        )


class MixPrecisionTransformerLayer(DeepSpeedTransformerLayer):
    """
    Compatibility:
    1. torch.autocast context manager supported
    2. support using after amp.initialize
    """

    def __init__(
        self,
        config,
        initial_weights=None,
        initial_biases=None,
        mix_precision=False,
    ):
        # create two layer function, so we can using both
        super(DeepSpeedTransformerLayer, self).__init__()

        self.config = config
        self.config.layer_id = DeepSpeedTransformerLayer.layer_id
        self.mix_precision = mix_precision
        if mix_precision:
            self.fp32_layer_id = DeepSpeedTransformerLayer.layer_id
            DeepSpeedTransformerLayer.layer_id = DeepSpeedTransformerLayer.layer_id + 1
            self.fp16_layer_id = DeepSpeedTransformerLayer.layer_id
            # self.config.layer_id = DeepSpeedTransformerLayer.layer_id
        DeepSpeedTransformerLayer.layer_id = DeepSpeedTransformerLayer.layer_id + 1

        if self.config.local_rank >= 0:
            torch.cuda.set_device(self.config.local_rank)
        self.attn_qkvw = nn.Parameter(torch.Tensor(self.config.hidden_size * 3, self.config.hidden_size))
        self.attn_qkvb = nn.Parameter(torch.Tensor(self.config.hidden_size * 3))
        self.attn_ow = nn.Parameter(torch.Tensor(self.config.hidden_size, self.config.hidden_size))
        self.attn_ob = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.attn_nw = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.attn_nb = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.inter_w = nn.Parameter(torch.Tensor(self.config.intermediate_size, self.config.hidden_size))
        self.inter_b = nn.Parameter(torch.Tensor(self.config.intermediate_size))
        self.output_w = nn.Parameter(torch.Tensor(self.config.hidden_size, self.config.intermediate_size))
        self.output_b = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.norm_w = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.norm_b = nn.Parameter(torch.Tensor(self.config.hidden_size))
        if initial_weights is None and initial_biases is None:
            self.init_transformer_weights(self.config.adjust_init_range)
        else:
            # For testing only.
            # copy data, otherwise tensor.data be released
            q = initial_weights[0].data
            k = initial_weights[1].data
            v = initial_weights[2].data

            self.attn_qkvw.data.copy_(torch.cat((q, k, v)))
            # copy bias
            self.attn_qkvb.data.copy_(
                torch.cat(
                    (
                        initial_biases[0].data,
                        initial_biases[1].data,
                        initial_biases[2].data,
                    )
                )
            )
            # self.attn_qkvb.data.zero_()

            self.attn_ow.data.copy_(initial_weights[3].data)
            self.attn_ob.data.copy_(initial_biases[3].data)
            self.attn_nw.data.copy_(initial_weights[4].data)
            self.attn_nb.data.copy_(initial_biases[4].data)
            self.inter_w.data.copy_(initial_weights[5].data)
            self.inter_b.data.copy_(initial_biases[5].data)
            self.output_w.data.copy_(initial_weights[6].data)
            self.output_b.data.copy_(initial_biases[6].data)
            self.norm_w.data.copy_(initial_weights[7].data)
            self.norm_b.data.copy_(initial_biases[7].data)
        # Load cuda modules if needed
        if transformer.transformer_cuda_module is None and not self.config.stochastic_mode:
            transformer.transformer_cuda_module = TransformerBuilder().load()
        if transformer.stochastic_transformer_cuda_module is None and self.config.stochastic_mode:
            transformer.stochastic_transformer_cuda_module = StochasticTransformerBuilder().load()

        # create the layer in cuda kernels.
        cuda_module = (
            transformer.stochastic_transformer_cuda_module
            if self.config.stochastic_mode
            else transformer.transformer_cuda_module
        )
        same_args = (
            self.config.batch_size,
            self.config.hidden_size,
            self.config.heads,
            self.config.intermediate_size,
            self.config.attn_dropout_ratio,
            self.config.hidden_dropout_ratio,
            self.config.layer_norm_eps,
            self.config.seed,
            self.config.pre_layer_norm,
            self.config.test_gemm,
            self.config.attn_dropout_checkpoint,
            self.config.normalize_invertible,
            self.config.gelu_checkpoint,
            self.config.stochastic_mode,
        )
        if mix_precision:
            cuda_module.create_transformer_layer_fp32(self.fp32_layer_id, *same_args)
            cuda_module.create_transformer_layer_fp16(self.fp16_layer_id, *same_args)
        else:
            create_layer_func = (
                cuda_module.create_transformer_layer_fp16
                if self.config.fp16
                else cuda_module.create_transformer_layer_fp32
            )
            create_layer_func(self.config.layer_id, *same_args)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        layer_head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        grads=None,
    ):
        self.config.is_grad_enabled = torch.is_grad_enabled()
        if self.mix_precision:
            if torch.is_autocast_enabled() or is_apex_amp_activate():
                self.config.fp16 = True
                self.config.layer_id = self.fp16_layer_id  # thread not safe
                args = (
                    hidden_states.half(),
                    attention_mask.half(),
                    self,
                    grads,
                    self.config.layer_id,
                    self.attn_qkvw.half(),
                    self.attn_qkvb.half(),
                    self.attn_ow.half(),
                    self.attn_ob.half(),
                    self.attn_nw.half(),
                    self.attn_nb.half(),
                    self.inter_w.half(),
                    self.inter_b.half(),
                    self.output_w.half(),
                    self.output_b.half(),
                    self.norm_w.half(),
                    self.norm_b.half(),
                    self.config,
                )
            else:
                self.config.fp16 = False
                self.config.layer_id = self.fp32_layer_id
                # when apex.amp enable, self.attn_qkvw is dtype of half
                args = (
                    hidden_states.float(),
                    attention_mask.float(),
                    self,
                    grads,
                    self.config.layer_id,
                    self.attn_qkvw.float(),
                    self.attn_qkvb.float(),
                    self.attn_ow.float(),
                    self.attn_ob.float(),
                    self.attn_nw.float(),
                    self.attn_nb.float(),
                    self.inter_w.float(),
                    self.inter_b.float(),
                    self.output_w.float(),
                    self.output_b.float(),
                    self.norm_w.float(),
                    self.norm_b.float(),
                    self.config,
                )  # will get float32 result
        else:
            args = (
                hidden_states,
                attention_mask,
                self,
                grads,
                self.config.layer_id,
                self.attn_qkvw,
                self.attn_qkvb,
                self.attn_ow,
                self.attn_ob,
                self.attn_nw,
                self.attn_nb,
                self.inter_w,
                self.inter_b,
                self.output_w,
                self.output_b,
                self.norm_w,
                self.norm_b,
                self.config,
            )
        ret = MixPrecisionDeepSpeedTransformerFunction.apply(*args)
        if self.mix_precision:
            if torch.is_autocast_enabled():
                ret_dtype = torch.float32
            elif is_apex_amp_activate():
                ret_dtype = _amp_state.opt_properties.cast_model_type
            else:
                ret_dtype = torch.float32
            if self.config.return_tuple:
                # outputs -> (output) : outputs[0] = output
                return tuple(o.to(ret_dtype) for o in ret)
            else:
                return ret.to(ret_dtype)
        return ret

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Keep parameter same as bert layer to support interchangeable read/write.
        sd = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        attn_qkvw = sd.pop(prefix + "attn_qkvw")
        attn_qkvb = sd.pop(prefix + "attn_qkvb")
        qw, kw, vw = attn_qkvw.split(attn_qkvw.size(0) // 3)
        qb, kb, vb = attn_qkvb.split(attn_qkvb.size(0) // 3)
        sd[prefix + "attention.self.query.weight"] = qw
        sd[prefix + "attention.self.query.bias"] = qb
        sd[prefix + "attention.self.key.weight"] = kw
        sd[prefix + "attention.self.key.bias"] = kb
        sd[prefix + "attention.self.value.weight"] = vw
        sd[prefix + "attention.self.value.bias"] = vb

        sd[prefix + "attention.output.dense.weight"] = sd.pop(prefix + "attn_ow")
        sd[prefix + "attention.output.dense.bias"] = sd.pop(prefix + "attn_ob")
        sd[prefix + "attention.output.LayerNorm.weight"] = sd.pop(prefix + "attn_nw")
        sd[prefix + "attention.output.LayerNorm.bias"] = sd.pop(prefix + "attn_nb")
        sd[prefix + "intermediate.dense.weight"] = sd.pop(prefix + "inter_w")
        sd[prefix + "intermediate.dense.bias"] = sd.pop(prefix + "inter_b")
        sd[prefix + "output.dense.weight"] = sd.pop(prefix + "output_w")
        sd[prefix + "output.dense.bias"] = sd.pop(prefix + "output_b")
        sd[prefix + "output.LayerNorm.weight"] = sd.pop(prefix + "norm_w")
        sd[prefix + "output.LayerNorm.bias"] = sd.pop(prefix + "norm_b")
        return sd

    def load_state_dict(self, state_dict, strict=True):
        # Support restore parameters from BertLayer parameters.

        qw = state_dict["attention.self.query.weight"]
        qb = state_dict["attention.self.query.bias"]
        kw = state_dict["attention.self.key.weight"]
        kb = state_dict["attention.self.key.bias"]
        vw = state_dict["attention.self.value.weight"]
        vb = state_dict["attention.self.value.bias"]
        attn_qkvw = torch.cat((qw, kw, vw))
        attn_qkvb = torch.cat((qb, kb, vb))
        new_state_dict = {
            "attn_qkvw": attn_qkvw,
            "attn_qkvb": attn_qkvb,
            "attn_ow": state_dict["attention.output.dense.weight"],
            "attn_ob": state_dict["attention.output.dense.bias"],
            "attn_nw": state_dict["attention.output.LayerNorm.weight"],
            "attn_nb": state_dict["attention.output.LayerNorm.bias"],
            "inter_w": state_dict["intermediate.dense.weight"],
            "inter_b": state_dict["intermediate.dense.bias"],
            "output_w": state_dict["output.dense.weight"],
            "output_b": state_dict["output.dense.bias"],
            "norm_w": state_dict["output.LayerNorm.weight"],
            "norm_b": state_dict["output.LayerNorm.bias"],
        }
        return super().load_state_dict(new_state_dict, strict=strict)


class BertAttentionFA(FlashMHA):
    """
    Compatible flash attention for HF BertAttention
    """

    _src_module_cls = BertAttention

    def __init__(self, config=None, src_module=None):
        mod_is_meta = False
        if src_module is None:
            logger.info("init without source module, use config or default")
            device = getattr(config, "device", None)
            dtype = getattr(config, "dtype", None)
            hidden_size = getattr(config, "hidden_size", 768)
            num_attention_heads = getattr(config, "num_attention_heads", 12)
            attention_probs_dropout_prob = getattr(config, "attention_probs_dropout_prob", 0.1)
            layer_norm_eps = getattr(config, "layer_norm_eps", 1e-12)
            hidden_dropout_prob = getattr(config, "hidden_dropout_prob", 0.1)
        else:
            assert isinstance(src_module, self._src_module_cls)
            mod_is_meta = is_meta(src_module)
            if mod_is_meta:
                reload_meta_module(src_module)
            # For arbitrarily usage location of apex(02) and .to(device),
            # just keep identical from the very beginning
            device = list(src_module.parameters())[0].device
            dtype = list(src_module.parameters())[0].dtype
            hidden_size = src_module.output.dense.in_features
            num_attention_heads = src_module.self.num_attention_heads
            attention_probs_dropout_prob = src_module.self.dropout.p
            layer_norm_eps = src_module.output.LayerNorm.eps
            hidden_dropout_prob = src_module.output.dropout.p
        super().__init__(
            hidden_size,
            num_attention_heads,
            bias=True,
            batch_first=True,
            attention_dropout=attention_probs_dropout_prob,
            causal=False,
            device=device,
            dtype=dtype,
        )
        # FlashMHA doesnt apply output dropout, skip connecting and output layernorm. Append them
        self.OutputLayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, device=device, dtype=dtype)
        self.OutputDropout = nn.Dropout(hidden_dropout_prob)
        self.register_lsd_hook()
        if src_module is not None:
            self.load_state_dict(src_module.state_dict())
            if mod_is_meta:
                recursive_empty_param(src_module, ignore_save=True)
                recursive_empty_param(self, prefix_name="replace_")
        if dropout_add_layer_norm is None:
            self.fuse_dropout_add_layer_norm = False
            logger.warning("dropout_add_layer_norm is not available, fall back to torch native api.")
        else:
            self.fuse_dropout_add_layer_norm = True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        past_key_value=None,  # 4.17
    ):
        """Compatible input/output for FlashMHA"""
        assert head_mask is None, "FlashMHA doesnt support head mask for now"
        assert (
            encoder_hidden_states is None and encoder_attention_mask is None
        ), "FlashMHA doesnt support cross-attention for now"
        assert output_attentions is False, "FlashMHA doesnt support output_attentions for now"
        if attention_mask is not None:
            if attention_mask.dtype is not torch.bool:  # [0., float('-inf')] -> [True, False]
                attention_mask = attention_mask == 0
            if len(attention_mask.shape) == 4:  # [bs, 1, 1, seq_len]
                attention_mask = attention_mask.squeeze(1).squeeze(1)
            assert (
                len(attention_mask.shape) == 2 and attention_mask.shape == hidden_states.shape[:2]
            ), "FlashMHA treats attn mask as key padding mask"
        assert (
            torch.is_autocast_enabled() or is_apex_amp_activate()
        ), "FlashMHA doesnt support FP32 for now, torch autocast or amp"

        attn_output, _ = super().forward(hidden_states, key_padding_mask=attention_mask)
        # FlashMHA doesnt apply output dropout, skip connecting and output layernorm. Append them
        if self.fuse_dropout_add_layer_norm:
            # [autocast] torch native LayerNorm excluded in autocast, causing increasing numerical error
            # compared with the fused one included in autocast. manually exclude the fused one for now.
            # [apex] Moreover, apex amp O2 computes in fp16 all through with model param in fp16,
            # we simply convert to ln weight dtype, so that it will be conducted in model param's dtype
            with torch.cuda.amp.autocast(enabled=False):  # torch 1.9.1 api, may differ from higher
                dtype = self.OutputLayerNorm.weight.dtype
                attn_output = dropout_add_layer_norm(
                    # attn_output.float(),
                    # hidden_states.float(),  # force input in fp32
                    attn_output.to(dtype),
                    hidden_states.to(dtype),
                    self.OutputLayerNorm.weight,
                    self.OutputLayerNorm.bias,
                    self.OutputDropout.p if self.training else 0.0,
                    self.OutputLayerNorm.eps,
                    rowscale=None,
                    prenorm=False,
                )
        else:
            attn_output = self.OutputDropout(attn_output)
            attn_output = self.OutputLayerNorm(attn_output + hidden_states)
        return (attn_output,)

    @staticmethod
    def transform_state_dict(sd):
        """func for manually transform directly saved state dict back to standard hf format,
        not overriding state_dict due to fsdp compat problem"""
        prefixes = {name[:-11] for name in sd.keys() if name.endswith("Wqkv.weight")}
        for prefix in prefixes:
            attn_qkvw = sd.pop(prefix + "Wqkv.weight")
            attn_qkvb = sd.pop(prefix + "Wqkv.bias")
            qw, kw, vw = attn_qkvw.split(attn_qkvw.size(0) // 3)
            qb, kb, vb = attn_qkvb.split(attn_qkvb.size(0) // 3)
            sd[prefix + "self.query.weight"] = qw
            sd[prefix + "self.query.bias"] = qb
            sd[prefix + "self.key.weight"] = kw
            sd[prefix + "self.key.bias"] = kb
            sd[prefix + "self.value.weight"] = vw
            sd[prefix + "self.value.bias"] = vb

            sd[prefix + "output.dense.weight"] = sd.pop(prefix + "out_proj.weight")
            sd[prefix + "output.dense.bias"] = sd.pop(prefix + "out_proj.bias")
            sd[prefix + "output.LayerNorm.weight"] = sd.pop(prefix + "OutputLayerNorm.weight")
            sd[prefix + "output.LayerNorm.bias"] = sd.pop(prefix + "OutputLayerNorm.bias")
        return sd

    def register_lsd_hook(self):
        """compat load weight from BertAttention"""

        def hook(state_dict, prefix, *args):
            try:
                qw = state_dict.pop(prefix + "self.query.weight")
                qb = state_dict.pop(prefix + "self.query.bias")
                kw = state_dict.pop(prefix + "self.key.weight")
                kb = state_dict.pop(prefix + "self.key.bias")
                vw = state_dict.pop(prefix + "self.value.weight")
                vb = state_dict.pop(prefix + "self.value.bias")
                attn_qkvw = torch.cat((qw, kw, vw))
                attn_qkvb = torch.cat((qb, kb, vb))
                state_dict[prefix + "Wqkv.weight"] = attn_qkvw
                state_dict[prefix + "Wqkv.bias"] = attn_qkvb

                state_dict[prefix + "out_proj.weight"] = state_dict.pop(prefix + "output.dense.weight")
                state_dict[prefix + "out_proj.bias"] = state_dict.pop(prefix + "output.dense.bias")
                state_dict[prefix + "OutputLayerNorm.weight"] = state_dict.pop(prefix + "output.LayerNorm.weight")
                state_dict[prefix + "OutputLayerNorm.bias"] = state_dict.pop(prefix + "output.LayerNorm.bias")
            except KeyError:
                logger.info("Not from standard hf BertAttention ckpt, assume from the not transformed state dict.")

        self._register_load_state_dict_pre_hook(hook)


class CLIPAttentionFA(FlashMHA):
    """
    Compatible flash attention for HF CLIPAttention
    """

    _src_module_cls = CLIPAttention

    def __init__(self, config=None, src_module=None):
        mod_is_meta = False
        if src_module is None:
            logger.info("init without source module, use config or default")
            device = getattr(config, "device", None)
            dtype = getattr(config, "dtype", None)
            hidden_size = getattr(config, "hidden_size", 512)
            num_attention_heads = getattr(config, "num_attention_heads", 8)
            attention_dropout = getattr(config, "attention_dropout", 0.0)
        else:
            assert isinstance(src_module, self._src_module_cls)
            mod_is_meta = is_meta(src_module)
            if mod_is_meta:
                reload_meta_module(src_module)
            # For arbitrarily usage location of apex(02) and .to(device),
            # just keep identical from the very beginning
            device = list(src_module.parameters())[0].device
            dtype = list(src_module.parameters())[0].dtype
            hidden_size = src_module.embed_dim
            num_attention_heads = src_module.num_heads
            attention_dropout = src_module.dropout
        super().__init__(
            hidden_size,
            num_attention_heads,
            bias=True,
            batch_first=True,
            attention_dropout=attention_dropout,
            causal=False,
            device=device,
            dtype=dtype,
        )
        self.register_lsd_hook()
        if src_module is not None:
            self.load_state_dict(src_module.state_dict())
            if mod_is_meta:
                recursive_empty_param(src_module, ignore_save=True)
                recursive_empty_param(self, prefix_name="replace_")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=False,
    ):
        """Compatible input/output for FlashMHA"""
        # if causal_attention_mask is not None, enable causal argument in FlashMHA
        self.causal = causal_attention_mask is not None
        assert output_attentions is False, "FlashMHA doesnt support output_attentions for now"
        if attention_mask is not None:
            if attention_mask.dtype is not torch.bool:  # [0., float('-inf')] -> [True, False]
                attention_mask = attention_mask == 0
            if len(attention_mask.shape) == 4:  # [bs, 1, tgt_len, seq_len]
                attention_mask = attention_mask[:, :, 0, :].squeeze(1).squeeze(1)
            assert (
                len(attention_mask.shape) == 2 and attention_mask.shape == hidden_states.shape[:2]
            ), "FlashMHA treats attn mask as key padding mask"
        assert (
            torch.is_autocast_enabled() or is_apex_amp_activate()
        ), "FlashMHA doesnt support FP32 for now, torch autocast or amp"

        return super().forward(hidden_states, key_padding_mask=attention_mask)

    @staticmethod
    def transform_state_dict(sd):
        """func for manually transform directly saved state dict back to standard hf format,
        not overriding state_dict due to fsdp compat problem"""
        prefixes = {name[:-11] for name in sd.keys() if name.endswith("Wqkv.weight")}
        for prefix in prefixes:
            attn_qkvw = sd.pop(prefix + "Wqkv.weight")
            attn_qkvb = sd.pop(prefix + "Wqkv.bias")
            qw, kw, vw = attn_qkvw.split(attn_qkvw.size(0) // 3)
            qb, kb, vb = attn_qkvb.split(attn_qkvb.size(0) // 3)
            sd[prefix + "q_proj.weight"] = qw
            sd[prefix + "q_proj.bias"] = qb
            sd[prefix + "k_proj.weight"] = kw
            sd[prefix + "k_proj.bias"] = kb
            sd[prefix + "v_proj.weight"] = vw
            sd[prefix + "v_proj.bias"] = vb
        return sd

    def register_lsd_hook(self):
        """compat load weight from CLIPAttention"""

        def hook(state_dict, prefix, *args):
            try:
                qw = state_dict.pop(prefix + "q_proj.weight")
                qb = state_dict.pop(prefix + "q_proj.bias")
                kw = state_dict.pop(prefix + "k_proj.weight")
                kb = state_dict.pop(prefix + "k_proj.bias")
                vw = state_dict.pop(prefix + "v_proj.weight")
                vb = state_dict.pop(prefix + "v_proj.bias")
                attn_qkvw = torch.cat((qw, kw, vw))
                attn_qkvb = torch.cat((qb, kb, vb))
                state_dict[prefix + "Wqkv.weight"] = attn_qkvw
                state_dict[prefix + "Wqkv.bias"] = attn_qkvb
            except KeyError:
                logger.info("Not from standard hf CLIPAttention ckpt, assume from the not transformed state dict.")

        self._register_load_state_dict_pre_hook(hook)


class MultiheadAttentionFA(FlashMHA):
    """compatible Flash Attention for torch.nn.MultiheadAttention"""

    _src_module_cls = MultiheadAttention

    def __init__(self, config=None, src_module=None):
        if src_module is None:
            logger.info("init without source module, use config or default")
            device = getattr(config, "device", None)
            dtype = getattr(config, "dtype", None)
            self.linear_has_bias = getattr(config, "linear_has_bias", True)
            self.batch_first = getattr(config, "batch_first", True)
            embed_dim = getattr(config, "embed_dim", 768)
            num_heads = getattr(config, "num_heads", 12)
            dropout = getattr(config, "dropout", 0.1)
        else:
            assert isinstance(src_module, self._src_module_cls)
            device = list(src_module.parameters())[0].device
            dtype = list(src_module.parameters())[0].dtype
            assert src_module._qkv_same_embed_dim, "qkv same dim in FlashMHA"
            assert not src_module.add_zero_attn, "FlashMHA doesnt support for now"
            assert src_module.bias_k is None and src_module.bias_v is None, "not add_bias_kv"
            self.linear_has_bias = src_module.in_proj_bias is not None
            self.batch_first = src_module.batch_first
            embed_dim, num_heads, dropout = src_module.embed_dim, src_module.num_heads, src_module.dropout
        if not self.batch_first:
            logger.warning("FlashMHA asserts batch_first. Consider switching to batch_first.")

        super().__init__(
            embed_dim,
            num_heads,
            bias=self.linear_has_bias,
            batch_first=True,  # force to be True in construction, handle input before forward.
            attention_dropout=dropout,
            causal=False,
            device=device,
            dtype=dtype,
        )

        self.register_lsd_hook()
        if src_module is not None:
            self.load_state_dict(src_module.state_dict())

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, is_causal=False):
        self.causal = is_causal
        assert attn_mask is None, "FlashMHA prefers key_padding_mask"

        # need_weights asserted False in FlashMHA
        need_weights = False
        # torch 2.0+ F._canonical_mask compat
        if key_padding_mask is not None:
            if torch.is_floating_point(key_padding_mask):
                key_padding_mask = key_padding_mask == 0
            elif key_padding_mask.dtype == torch.bool:
                # key_padding_mask bool value in opposite
                key_padding_mask = ~key_padding_mask
            else:
                raise ValueError(f"key_padding_mask in {key_padding_mask.dtype}, not float/bool")

        if not self.batch_first:
            query = query.transpose(1, 0)
        attn_output, _ = super().forward(query, key_padding_mask=key_padding_mask, need_weights=need_weights)
        if not self.batch_first:
            attn_output = attn_output.transpose(1, 0)
        return attn_output, _

    @staticmethod
    def transform_state_dict(sd):
        """func for manually transform directly saved state dict back to standard nnMHA format,
        not overriding state_dict due to fsdp compat problem"""
        prefixes = {name[:-11] for name in sd.keys() if name.endswith("Wqkv.weight")}
        for prefix in prefixes:
            attn_qkvw = sd.pop(prefix + "Wqkv.weight")
            sd[prefix + "in_proj_weight"] = attn_qkvw
            try:
                attn_qkvb = sd.pop(prefix + "Wqkv.bias")
                sd[prefix + "in_proj_bias"] = attn_qkvb
            except KeyError:
                logger.info("Not linear_has_bias.")
        return sd

    def register_lsd_hook(self):
        """compat load weight from torch.nn.MultiheadAttention"""

        def hook(state_dict, prefix, *args):
            try:
                state_dict[prefix + "Wqkv.weight"] = state_dict.pop(prefix + "in_proj_weight")
                if self.linear_has_bias:
                    state_dict[prefix + "Wqkv.bias"] = state_dict.pop(prefix + "in_proj_bias")
            except KeyError:
                logger.info(
                    "Not from standard torch.nn.MultiheadAttention ckpt, assume from the not transformed state dict."
                )

        self._register_load_state_dict_pre_hook(hook)


def flash_attn_with_mask_bias(q, k, v, mask=None, bias=None, dropout_p=0.0, softmax_scale=None, causal=False):
    """
    FlashAttention that support mask. dropout_p should be set to 0.0 during evaluation.
    q, k, v, mask should be half precision(torch.float16 or torch.bfloat16).

    We use the following notation:
        b: batch_size
        s_q, s_k: sequence length of Q and K
        nh: number of attention heads
        hs: head dimension

    Args:
        q: [b, s_q, nh, hs]
        k/v: [b, s_k, nh, hs]
        mask: [b, nh or 1, s_q or 1, s_k]
        bias: [1, nh, s_q, s_k]  # not supported yet
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(hs).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).

    Returns:
        out:[b, s_q, nh, hs]
    """
    if bias is not None:
        raise NotImplementedError("FlashAttention1 does not support bias.")
    if has_legacy_fa1:
        _flash_attn_unpadded_func = flash_attn_1_unpadded_func
    else:
        _flash_attn_unpadded_func = flash_attn_unpadded_func

    b, s_q, nh, hs = q.shape
    _, s_k, _, _ = k.shape

    # work around odd seq_k https://github.com/HazyResearch/flash-attention/pull/57 @robotcator
    # and it's found that seq_q % 8 != 0 will lead to occasionally nan in k.grad/v.grad
    padded_s_q = False
    if s_q % 8 != 0:
        padded_s_q = True
        pad_len = 8 - s_q % 8
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
        s_q = s_q + pad_len
        if mask is not None and mask.shape[2] != 1:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, pad_len), value=0.0)
        if bias is not None:
            bias = torch.nn.functional.pad(bias, (0, 0, 0, pad_len), value=0.0)

    if s_k % 2 == 1:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, 1))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, 1))
        if mask is not None:
            mask = torch.nn.functional.pad(mask, (0, 1), value=-65504.0)
        if bias is not None:
            bias = torch.nn.functional.pad(bias, (0, 1), value=-65504.0)
        s_k = s_k + 1

    q = q.reshape(b * s_q, nh, hs)
    k = k.reshape(b * s_k, nh, hs)
    v = v.reshape(b * s_k, nh, hs)
    q_max_s = s_q
    q_cu_seqlens = torch.arange(0, (b + 1) * s_q, step=s_q, dtype=torch.int32, device=q.device)
    k_max_s = s_k
    k_cu_seqlens = torch.arange(0, (b + 1) * s_k, step=s_k, dtype=torch.int32, device=q.device)
    kwargs = {
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
    }
    if mask is not None or bias is not None:
        assert (
            is_additive_mask_bias_supported_fa1() or has_legacy_fa1
        ), "Must be attn mask/bias supported version FlashAttn v1"
        kwargs.update({"attn_mask": mask, "attn_bias": bias})
    out = _flash_attn_unpadded_func(
        q,
        k,
        v,
        q_cu_seqlens,
        k_cu_seqlens,
        q_max_s,
        k_max_s,
        **kwargs,
    )
    out = out.reshape(b, s_q, nh, hs)
    if padded_s_q:
        out = out[:, :-pad_len, :, :]
    return out


def fa2_with_glm_mask(q, k, v, glm_mask=None, dropout_p=0.0, softmax_scale=None, causal=True):
    """
    FA2 that support glm-style mask. dropout_p should be set to 0.0 during evaluation.
    Only support half precision(torch.float16 or torch.bfloat16).

    Args:
        glm_mask: [batch_size] in torch.int32.
        other args: see `flash_attn_with_mask_bias` or `flash_attn_func` in FA2
    """
    if glm_mask is not None:
        assert causal, "causal must be True for glm_mask"
    kwargs = {
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "return_attn_probs": False,
    }
    if glm_mask is not None:
        assert is_glm_mask_supported_fa2(), "please install glm mask supported version FA2"
        kwargs.update({"glm_mask": glm_mask})
    return flash_attn_func(q, k, v, **kwargs)


class FlashAttnModule(nn.Module):
    """
    Unified API for FlashAttention, make it a module for train/val dropout_p handling.
    support key_padding_mask; support additive mask/bias in FA1; support break_point index glm-mask in FA2

    Update:: (2.3.6+pack.glm.mask) support startpoint/endpoint index pack glm_mask in torch.int32 dtype and
    [batch_size, 2, MAX_NUM_PAIR] shape, compat [batch_size] break_point index glm-mask.

    Note:: compat for single q with past key value in decoding, as atorch fa version does not include
           q/k uneven causal fix in https://github.com/Dao-AILab/flash-attention/issues/282.
    """

    npu_cache = {"breakpoint_additive_mask": None, "breakpoint_prefix": None}

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.attention_dropout = attention_dropout

    def forward(self, q, k, v, key_padding_mask=None, glm_mask=None, additive_mask=None, additive_bias=None):
        """
        key_padding_mask: a bool tensor of shape (B, S), bool / int, True / 1 means valid and 0 means not valid.
        """
        b, s_q, nh, hs = q.shape
        _, s_k, _, _ = k.shape

        kwargs = {
            "dropout_p": self.attention_dropout if self.training else 0.0,
            "softmax_scale": self.softmax_scale,
        }

        if s_q == 1:
            # for single q decoding, not use causal
            kwargs.update({"causal": False})
        else:
            assert (
                s_q == s_k or not self.causal
            ), f"Support single query decoding for now, query seq len {s_q}, key seq len {s_k}."
            kwargs.update({"causal": self.causal})

        # FA2 glm_mask
        if glm_mask is not None:
            assert self.causal, "causal must be True for glm_mask"
            assert is_glm_mask_supported_fa2(), "please install glm mask supported version FA2"
            kwargs.update({"glm_mask": glm_mask})

            if is_torch_npu_available():
                kwargs.update(self.npu_cache)
                return npu_fa_with_glm_mask(q, k, v, **kwargs)

        # FA1 additive mask
        if additive_mask is not None or additive_bias is not None:
            assert (
                is_additive_mask_bias_supported_fa1() or has_legacy_fa1
            ), "Must be attn mask/bias supported version FlashAttn v1"
            assert not self.causal and key_padding_mask is None, "Should not causal/padding mask when additive mask"
            if is_torch_npu_available():
                assert additive_bias is None, "HPU FlashAttenion does not suppport additive_bias"
                kwargs.update({"glm_mask": additive_mask})
                return npu_fa_with_glm_mask(q, k, v, **kwargs)
            kwargs.update({"mask": additive_mask, "bias": additive_bias})

        if key_padding_mask is None or key_padding_mask.bool().all():
            if is_xla_device_available():
                return xla_flash_attn(
                    q,
                    k,
                    v,
                    dropout_rate=kwargs["dropout_p"],
                    scale=kwargs["softmax_scale"],
                    is_causal=kwargs["causal"],
                )
            elif _flash_attn_version >= parse_version("2") and "mask" not in kwargs and "bias" not in kwargs:
                return flash_attn_func(q, k, v, **kwargs)
            else:
                return flash_attn_with_mask_bias(q, k, v, **kwargs)
        else:
            # unpad input and pad output according key_padding_mask
            # Note: [2023-12-08] single q decoding, only the last mask meaningful to query
            q_unpad, indices, cu_seqlens, max_seqlen = unpad_input(
                q, key_padding_mask[:, -1:] if s_q == 1 else key_padding_mask
            )
            k_unpad, k_indices, k_cu_seqlens, k_max_seqlen = unpad_input(k, key_padding_mask)
            v_unpad, _, _, _ = unpad_input(v, key_padding_mask)

            if is_xla_device_available():
                o_unpad = xla_flash_attn_varlen(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    cu_seqlens_query=cu_seqlens,
                    cu_seqlens_key=k_cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=k_max_seqlen,
                    dropout_rate=kwargs["dropout_p"],
                    scale=kwargs["softmax_scale"],
                    is_causal=kwargs["causal"],
                )
            else:
                o_unpad = flash_attn_unpadded_func(
                    q_unpad, k_unpad, v_unpad, cu_seqlens, k_cu_seqlens, max_seqlen, k_max_seqlen, **kwargs
                )
            return pad_input(o_unpad, indices, b, s_q)


class LlamaAttentionFA(LlamaAttention):
    def __new__(cls, src_module):
        assert src_module is not None and isinstance(src_module, LlamaAttention)
        from atorch.utils.meta_model_utils import custom_transform_model_keep_checkpoint_name

        new_module = custom_transform_model_keep_checkpoint_name(src_module, lambda x: copy.deepcopy(x))
        new_module.__class__ = cls
        return new_module

    def __init__(self, *args, **kwargs):
        self.FA = FlashAttnModule(causal=True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        ###### FA Compute #######
        if len(attention_mask.shape) == 4:
            # FA note: llama pre-add causal mask and padding mask, convert back to padding mask
            key_padding_mask = attention_mask[:, 0, -1, :] == 0
        elif len(attention_mask.shape) == 2:
            # if attention_mask is not processed and remained key padding mask format
            key_padding_mask = attention_mask
        else:
            raise ValueError(f"Attention mask format not supported, {attention_mask.dtype}, {attention_mask.shape}")
        attn_output = self.FA(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            key_padding_mask=key_padding_mask,
        )
        ###### FA Compute End ###

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class HFGLMSelfAttentionFA(torch.nn.Module):
    """SelfAttention layer optimized by flash-attention for HuggingFace GLM.
    source: https://huggingface.co/THUDM/glm-10b/blob/main/modeling_glm.py#L213

    Self-attention layer takes input with size [b, s, h] where b is
    the batch size, s is the sequence lenght, and h is the hidden size
    and creates output of the same size.
    Arguments:
        hidden_size: total hidden size of the layer (h).
        num_attention_heads: number of attention heads (n). Note that we
                             require n to be divisible by number of GPUs
                             used to parallelize the model. Also, we
                             require hidden size to be divisible by n.
        attention_dropout_prob: dropout probability for the attention scores.
        init_method: weight initialization.
        output_layer_init_method: output layer initialization. If None, use
                                  `init_method`.
    We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
    """

    def __init__(self, config=None, src_module=None):
        super(HFGLMSelfAttentionFA, self).__init__()

        if src_module is None:
            # Per attention head and per partition values.
            self.hidden_size = getattr(config, "hidden_size", 4096)
            self.num_attention_heads = getattr(config, "num_attention_heads", 64)
            self.hidden_size_per_attention_head = divide(self.hidden_size, self.num_attention_heads)
            self.attention_scale = getattr(config, "attention_scale", 1.0)
            # Strided linear layer.
            self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size)

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            attention_dropout_prob = getattr(config, "attention_dropout_prob", 0.1)
            self.attention_dropout = nn.Dropout(attention_dropout_prob)

            # Output.
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            output_dropout_prob = getattr(config, "output_dropout_prob", 0.1)
            self.output_dropout = nn.Dropout(output_dropout_prob)
        else:
            mod_is_meta = is_meta(src_module)
            if mod_is_meta:
                reload_meta_module(src_module)
            self.hidden_size = src_module.hidden_size
            self.hidden_size_per_attention_head = src_module.hidden_size_per_attention_head
            self.num_attention_heads = src_module.num_attention_heads
            self.attention_scale = src_module.attention_scale
            self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size)
            attention_dropout_prob = src_module.attention_dropout.p
            self.attention_dropout = nn.Dropout(attention_dropout_prob)
            self.dense = nn.Linear(self.hidden_size, self.hidden_size)
            output_dropout_prob = src_module.output_dropout.p
            self.output_dropout = torch.nn.Dropout(output_dropout_prob)
        if src_module is not None:
            self.load_state_dict(src_module.state_dict())
            if mod_is_meta:
                recursive_empty_param(src_module, ignore_save=True)
                recursive_empty_param(self, prefix_name="replace_")

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ltor_mask, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [b,1,s,s]

        # Attention heads. [b, s, hp]
        query_length = hidden_states.size(1)
        # self attention
        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer, mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            cat = torch.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer, mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]

        # Reshape and transpose [b, s, np, hn]
        query_layer = mixed_query_layer.view(
            *mixed_query_layer.size()[:-1], self.num_attention_heads, self.hidden_size_per_attention_head
        )
        key_layer = mixed_key_layer.view(
            *mixed_key_layer.size()[:-1], self.num_attention_heads, self.hidden_size_per_attention_head
        )
        value_layer = mixed_value_layer.view(
            *mixed_value_layer.size()[:-1], self.num_attention_heads, self.hidden_size_per_attention_head
        )
        if ltor_mask.dtype != query_layer.dtype:
            ltor_mask = ltor_mask.to(query_layer.dtype)
        dropout_p = self.attention_dropout.p if self.training else 0.0

        context_layer = flash_attn_with_mask_bias(
            query_layer, key_layer, value_layer, mask=(-65504.0) * (1.0 - ltor_mask), dropout_p=dropout_p
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)
        return output


class GPT2AttentionFA(GPT2Attention):
    def __new__(self, src_module=None):
        assert src_module is not None and isinstance(src_module, GPT2Attention)
        new_module = copy.deepcopy(src_module)
        new_module.__class__ = GPT2AttentionFA
        return new_module

    def __init__(self, *args, **kwargs):
        pass

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        assert head_mask is None, "currently not handling head_mask"
        assert not output_attentions, "currently not output attentions"
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # [b, s_q/s_k, nh, hs]
        query = query.view(query.shape[:-1] + (self.num_heads, self.head_dim))
        key = key.view(key.shape[:-1] + (self.num_heads, self.head_dim))
        value = value.view(value.shape[:-1] + (self.num_heads, self.head_dim))

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=1)  # [b, s_k, nh, hs]
            value = torch.cat((past_value, value), dim=1)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # softmax scale
        softmax_scale = 1.0
        if self.scale_attn_weights:
            softmax_scale = softmax_scale / (value.size(-1) ** 0.5)
        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            softmax_scale = softmax_scale / float(self.layer_idx + 1)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        if attention_mask is not None:
            attention_mask = attention_mask.to(query.dtype)
        attn_output = flash_attn_with_mask_bias(
            query, key, value, mask=attention_mask, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=True
        )
        attn_output = attn_output.view(attn_output.size()[:-2] + (self.num_heads * self.head_dim,))

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs  # a, present, (attentions)
