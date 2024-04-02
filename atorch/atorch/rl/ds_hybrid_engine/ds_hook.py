import math
import os

import torch
import torch.nn.functional as F
from deepspeed import comm as dist
from deepspeed.inference.config import DeepSpeedInferenceConfig  # NOQA
from deepspeed.model_implementations.transformers.ds_transformer import DeepSpeedTransformerInference  # NOQA
from deepspeed.module_inject import ReplaceWithTensorSlicing
from deepspeed.module_inject.containers.base import BaseTransformerContainer
from deepspeed.module_inject.containers.features import HybridSplitQKVContainer
from deepspeed.module_inject.containers.features.gated_mlp import HybridGatedMLPContainer
from deepspeed.module_inject.containers.features.hybrid_engine import HybridEngineContainer
from deepspeed.ops.transformer.inference.ds_attention import DeepSpeedSelfAttention
from deepspeed.ops.transformer.inference.ds_mlp import DeepSpeedMLP
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv


def mlp_inter_mp(self, mp_replace, reversed_dim=False):
    # Only need to alter behavior if we can't do the normal destructive copy
    if self.module.mlp.inter_w is None:

        reversed_dim = True
        self.module.mlp.inter_up_w = mp_replace.copy(
            self.module.mlp.inter_up_w, self.inter_up_w, int8=reversed_dim, allocate_tensor=reversed_dim
        )

        self.module.mlp.inter_gate_w = mp_replace.copy(
            self.module.mlp.inter_gate_w, self.inter_gate_w, int8=reversed_dim, allocate_tensor=reversed_dim
        )

    else:
        self.module.mlp.inter_w = mp_replace.strided_copy(
            self.module.mlp.inter_w, self._h4h_w, num_splits=2, int8=reversed_dim
        )
        self.module.mlp.inter_b = mp_replace.strided_copy(
            self.module.mlp.inter_b, self._h4h_b, num_splits=2, int8=reversed_dim
        )


HybridGatedMLPContainer.mlp_inter_mp = mlp_inter_mp


def attention_qkv_mp_hybrid(self, mp_replace, reversed_dim=False):
    # Only need to alter
    if self.module.attention.attn_qkvw is None:
        # self.module.attention.attn_qw type is torch.nn.Parameter
        reversed_dim = True
        self.module.attention.attn_qw = mp_replace.copy(self.module.attention.attn_qw, self.qw, int8=reversed_dim)
        if self.qb is not None:
            self.module.attention.attn_qb = mp_replace.copy(self.module.attention.attn_qb, self.qb, int8=reversed_dim)
        else:
            self.module.attention.attn_qb = None
        self.module.attention.attn_kw = mp_replace.copy(self.module.attention.attn_kw, self.kw, int8=reversed_dim)

        if self.kb is not None:
            self.module.attention.attn_kb = mp_replace.copy(self.module.attention.attn_kb, self.kb, int8=reversed_dim)
        else:
            self.module.attention.attn_kb = None

        self.module.attention.attn_vw = mp_replace.copy(self.module.attention.attn_vw, self.vw, int8=reversed_dim)

        if self.vb is not None:
            self.module.attention.attn_vb = mp_replace.copy(self.module.attention.attn_vb, self.vb, int8=reversed_dim)
        else:
            self.module.attention.attn_vb = None

    else:
        super(HybridEngineContainer, self).attention_qkv_mp(mp_replace)


HybridSplitQKVContainer.attention_qkv_mp = attention_qkv_mp_hybrid


def attention_qkv_mp(self, mp_replace, reversed_dim=False):
    reversed_dim = True
    self.module.attention.attn_qkvw = mp_replace.strided_copy(
        self.module.attention.attn_qkvw, self.qkvw, num_splits=3, int8=reversed_dim
    )
    reversed_dim = True
    self.module.attention.attn_qkvb = mp_replace.strided_copy(
        self.module.attention.attn_qkvb, self.qkvb, num_splits=3, int8=reversed_dim
    )


def attention_o_mp(self, mp_replace, reversed_dim=False):
    reversed_dim = False
    self.module.attention.attn_ow = mp_replace.copy(self.module.attention.attn_ow, self.dense_w, int8=reversed_dim)
    if self.dense_b is not None:
        self.module.attention.attn_ob = torch.nn.Parameter(self.dense_b.clone())
    else:
        self.module.attention.attn_ob = None


def mlp_inter_mp_base(self, mp_replace, reversed_dim=False):
    reversed_dim = True
    self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w, self._h4h_w, int8=reversed_dim)
    if self._h4h_b is not None:
        self.module.mlp.inter_b = mp_replace.copy(self.module.mlp.inter_b, self._h4h_b, int8=reversed_dim)
    else:
        self._h4h_b = None


def mlp_output_mp(self, mp_replace, reversed_dim=False):
    reversed_dim = False
    self.module.mlp.output_w = mp_replace.copy(self.module.mlp.output_w, self._4hh_w, int8=reversed_dim)
    if self._4hh_b is not None:
        self.module.mlp.output_b = torch.nn.Parameter(self._4hh_b.clone())
    else:
        self.module.mlp.output_b = None


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value.
    """
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    return numerator // denominator


def strided_copy(self, dst, src, num_splits: int, int8: bool = False, allocate_tensor: bool = False):
    rank = self.gpu_index
    world_size = self.mp_size

    shape = src.shape
    if int8:
        per_partition_size = divide(shape[0], world_size)
        int8 = False
    else:
        int8 = True
        per_partition_size = divide(shape[1], world_size)

    # clone master weight, becase master weight
    # would be partitioned
    # in hybrid engine, master weight is not None
    src = torch.clone(src)

    if world_size == 1:
        return torch.nn.Parameter(src)

    stride = num_splits
    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(src, per_partition_per_stride_size, int8)
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        dst = torch.nn.Parameter(torch.cat(my_weight_list, int8))
    return dst


def copy(self, dst, src, int8=False, allocate_tensor=False):
    # 如果是按行切分
    rank = self.gpu_index
    world_size = self.mp_size

    shape = src.shape
    if int8:
        per_partition_size = divide(shape[0], world_size)
        int8 = 0
    else:
        int8 = 1
        per_partition_size = divide(shape[1], world_size)

    # clone master weight, becase master weight
    # would be partitioned
    # in hybrid engine, master weight is not None
    src = torch.clone(src)

    if world_size == 1:
        return torch.nn.Parameter(src)

    stride = 1
    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)

    weight_list = torch.split(src, per_partition_per_stride_size, int8)
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        dst = torch.nn.Parameter(torch.cat(my_weight_list, int8))
    return dst


# hook deepspeed's function
BaseTransformerContainer.attention_o_mp = attention_o_mp
BaseTransformerContainer.mlp_inter_mp = mlp_inter_mp_base
BaseTransformerContainer.mlp_output_mp = mlp_output_mp
BaseTransformerContainer.attention_qkv_mp = attention_qkv_mp


ReplaceWithTensorSlicing.copy = copy
ReplaceWithTensorSlicing.strided_copy = strided_copy


def llama2_mlp_forward(self, attention_output, input, inp_norm, bias):
    gate = F.linear(attention_output, self.inter_gate_w)
    up = F.linear(attention_output, self.inter_up_w)
    gate_up = F.silu(gate) * up

    outputs = F.linear(gate_up, self.output_w)
    if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
        dist.all_reduce(outputs, group=self.mp_group)
    return outputs


def llama2_decoder_layer_forward(
    self,
    input=None,
    input_mask=None,
    attention_mask=None,
    attn_mask=None,
    head_mask=None,
    layer_past=None,
    get_key_value=False,
    get_present=False,
    encoder_output=None,
    enc_dec_attn_mask=None,
    x=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    use_cache=False,
    alibi=None,
    output_attentions=False,
    # TODO(arashb): 'layer_head_mask' and 'past_key_value' are only added to satisfy the OPT models API.
    # This needs to be redesigned later!
    layer_head_mask=None,
    past_key_value=None,
    **kwargs,
):

    head_mask = input_mask
    if x is not None:
        input = x
    if "hidden_states" in kwargs:
        input = kwargs["hidden_states"]

    input_mask = (input_mask if attn_mask is None else attn_mask) if attention_mask is None else attention_mask

    head_mask = kwargs["position_ids"]

    get_present = get_present or get_key_value or use_cache
    input_mask = input_mask if attention_mask is None else attention_mask

    # We set the prev key/value to None when there is a prompt
    if input.shape[1] > 1:
        self.layer_past = None
    # layer_past = layer_past if layer_past is not None else self.layer_past
    layer_past = past_key_value

    head_mask = layer_head_mask if layer_head_mask is not None else head_mask

    attn_mask = None
    if isinstance(input, tuple):
        attn_mask = input[1]
        input = input[0]

    # RMS NORM

    residual = input
    input_dtype = input.dtype
    hidden_states = input.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
    hidden_states = self.norm_w * hidden_states.to(input_dtype)

    with torch.no_grad():
        attention_output, past_key_value, _, context_outputtn_ctx, inp_norm = self.attention(
            hidden_states,
            input_mask,
            head_mask,
            layer_past,
            get_present,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            self.norm_w,
            self.norm_b,
            alibi,
        )
    hidden_states = residual + attention_output
    residual = hidden_states

    # RMS NORM
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
    hidden_states = self.mlp.attn_nw * hidden_states.to(input_dtype)

    hidden_states = self.mlp(hidden_states, None, None, None)
    hidden_states = residual + hidden_states

    return hidden_states, past_key_value, None


def llama2_attention_forward(
    self,
    input,
    input_mask,
    head_mask=None,
    layer_past=None,
    get_present=False,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=False,
    norm_w=None,
    norm_b=None,
    alibi=None,
):
    """
    Forward pass of the attention module.

    Args:
        x (torch.Tensor): Input tensor.
        start_pos (int): Starting position for caching.
        freqs_cis (torch.Tensor): Precomputed frequency tensor.
        mask (torch.Tensor, optional): Attention mask tensor.

    Returns:
        torch.Tensor: Output tensor after attention.

    """

    bsz, q_len, _ = input.size()

    hidden_states = input
    position_ids = head_mask
    attention_mask = input_mask
    # query_states = self.q_proj(hidden_states)
    query_states = F.linear(hidden_states, self.attn_qw)
    # key_states = self.k_proj(hidden_states)
    key_states = F.linear(hidden_states, self.attn_kw)
    # value_states = self.v_proj(hidden_states)
    value_states = F.linear(hidden_states, self.attn_vw)

    self.num_heads = self.config.llama.num_attention_heads
    self.num_key_value_heads = self.config.llama.num_key_value_heads
    self.head_dim = self.config.hidden_size // self.num_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    # for tp

    self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
    self.hidden_size_per_attention_head = self.config.hidden_size // self.num_heads
    self.num_attention_heads_per_partition = self.num_heads // self.config.mp_size
    self.num_key_value_heads_per_partition = self.config.llama.num_key_value_heads // self.config.mp_size

    if not hasattr(self, "rotary_emb"):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim, max_position_embeddings=self.config.llama.max_position_embeddings
        ).to(int(os.environ.get("LOCAL_RANK", 0)))

    past_key_value = layer_past

    query_states = query_states.view(bsz, q_len, self.num_attention_heads_per_partition, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads_per_partition, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads_per_partition, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_attention_heads_per_partition, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size"
            f" {(bsz,self.num_attention_heads_per_partition , q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + input_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_attention_heads_per_partition, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size"
            f" {(bsz, self.num_attention_heads_per_partition, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size_per_partition)

    attn_output = F.linear(attn_output, self.attn_ow)

    if self.mp_group is not None and dist.get_world_size(group=self.mp_group) > 1:
        dist.all_reduce(attn_output, group=self.mp_group)

    if not output_attentions:
        attn_weights = None
    return attn_output, past_key_value, _, None, None


# DeepSpeedTransformerInference.forward = opt_decoder_layer_forward
# glm_block_forward
DeepSpeedTransformerInference.forward = llama2_decoder_layer_forward
DeepSpeedSelfAttention.forward = llama2_attention_forward
DeepSpeedMLP.forward = llama2_mlp_forward
