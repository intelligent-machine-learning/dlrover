# This registry resgisters all shardable operators and the corresponding sharded implementations

# Global variables:
#     _SHARDABLE_OPERATORS (dict): maps a unique identifier to the operator
#     _SHARDED_OPERATORS (dict): maps a unique identifier to the sharded operator
#     _SHARD_MAP (dict): maps shardable operators to the corresponding sharded implementations
#         uses encodings specified in _SHARDABLE_OPERATORS and _SHARDED_OPERATORS
#     _REPLACE_SPECS (dict): maps a sharded operator to a callable. This callable should
#         specify the sharding specs for the input/output tensors, given the
#         default sharding specs of the original operator. All tensors are default to DDP sharding
#     _LOCAL_OPERATORS (list): a list of "local" operators. These operators perform element-wise
#         operations and thus works for all shardings of a tensor
#     _MAGIC_LOCAL_OPERATORS (list): a list of "local" operators. These operators are specified with
#         there class names, including +, -, etc.
#     To implement a tensor parallel operator that is supported by auto parallel optimization,
#         the operator must be registered here
import operator

import numpy as np
import torch
import transformers
from torch.fx.passes.shape_prop import TensorMetadata
from transformers.models.bert.modeling_bert import BertAttention, BertSelfAttention, BertSelfOutput
from transformers.models.clip.modeling_clip import CLIPMLP, CLIPAttention

from atorch.common.log_utils import default_logger as logger
from atorch.utils.graph_transform_utils import map_aggregate
from atorch.utils.meta_overrides import (
    _DEVICE,
    attention_func,
    mlp_func,
    register_meta_overrides,
    transformer_block_func,
)
from atorch.utils.sharding_spec import MeshShardingSpec

from .layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from .transformer import (
    ColumnParallelBertSelfAttention,
    MegatronBertAttention,
    MegatronCLIPAttention,
    MegatronCLIPMLP,
    MegatronGLMBlock,
    MegatronGLMMLP,
    MegatronGLMSelfAttention,
    MegatronGPTNeoXAttention,
    MegatronGPTNeoXLayer,
    MegatronGPTNeoXMLP,
    RowParallelBertSelfOutput,
)
from .utils import _compute_tensor_size

# -----------------
# Hack non tracable functions
# -----------------


torch.fx.wrap("len")


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


transformers.models.clip.modeling_clip.contrastive_loss = contrastive_loss

# -----------------
# Defining input/output sharding specification
# -----------------


def _empty_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    def _rewrite_spec(sharding_spec):
        if isinstance(sharding_spec, MeshShardingSpec):
            sharding_spec.dims = tuple()
            sharding_spec.group = group
            sharding_spec.ranks = ranks
            return sharding_spec
        else:
            return sharding_spec

    input_spec = map_aggregate(old_input_spec, _rewrite_spec)
    output_spec = map_aggregate(old_output_spec, _rewrite_spec)
    return input_spec, output_spec


def _row_linear_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    """Transform the original input/output spec into the row linear form
    Assign the correct group to the specs

    Args:
        old_input_spec is a dict, with keys: input_node name, value: sharding_spec (tuple/list/dict)
        old_output_spec is simply a sharding spec (or a tuple/list/dict of sharding specs)

    Returns:
        input_spec and output_spec for RowParallelLinear module node.
    """

    if isinstance(old_input_spec, dict):
        for key, sharding_spec in old_input_spec.items():
            # For linear layer, there should only be one input.
            # This input should be a Tensor, so we can
            # get directly the sharding spec
            if isinstance(sharding_spec, MeshShardingSpec):
                sharding_spec.dims = tuple([-1])
                sharding_spec.group = group
                sharding_spec.ranks = ranks
                old_input_spec[key] = sharding_spec
    if isinstance(old_output_spec, MeshShardingSpec):
        old_output_spec.dims = tuple()
        old_output_spec.group = group
        old_output_spec.ranks = ranks
    else:
        raise ValueError("Currently only MeshShardingSpec is supported")
    return old_input_spec, old_output_spec


def _col_linear_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    if isinstance(old_input_spec, dict):
        for key, sharding_spec in old_input_spec.items():
            # For linear layer, there should only be one input.
            # This input should be a Tensor, so we can
            # get directly the sharding spec
            if isinstance(sharding_spec, MeshShardingSpec):
                sharding_spec.dims = tuple()
                sharding_spec.group = group
                sharding_spec.ranks = ranks
                old_input_spec[key] = sharding_spec

    if isinstance(old_output_spec, MeshShardingSpec):
        old_output_spec.dims = tuple([-1])
        old_output_spec.group = group
        old_output_spec.ranks = ranks
    else:
        raise ValueError("Currently only MeshShardingSpec is supported")
    return old_input_spec, old_output_spec


def _col_bert_self_attn_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    """Transform the original input/output spec into the column parallel bert self attention form
    Assign the correct group to the specs

    Args:
        old_input_spec is a dict, with keys: input_node name, value: sharding_spec (tuple/list/dict)
        old_output_spec is simply a sharding spec (or a tuple/list/dict of sharding specs)

    Returns:
        input_spec and output_spec for RowParallelLinear module node.
    """

    if isinstance(old_input_spec, dict):
        for key, sharding_spec in old_input_spec.items():
            # The input of column bert self attention are not partitioned
            if isinstance(sharding_spec, MeshShardingSpec):
                sharding_spec.dims = tuple()
                sharding_spec.group = group
                sharding_spec.ranks = ranks
                old_input_spec[key] = sharding_spec

    if isinstance(old_output_spec, MeshShardingSpec):
        # last dimension is sharded
        old_output_spec.dims = tuple([-1])
        old_output_spec.group = group
        old_output_spec.ranks = ranks
    else:
        raise ValueError("Currently only MeshShardingSpec is supported")
    return old_input_spec, old_output_spec


def _row_bert_self_output_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    """Transform the original input/output spec into the row parallel self output form
    Assign the correct group to the specs

    Args:
        old_input_spec is a dict, with keys: input_node name, value: sharding_spec (tuple/list/dict)
        old_output_spec is simply a sharding spec (or a tuple/list/dict of sharding specs)

    Returns:
        input_spec and output_spec for RowParallelLinear module node.
    """

    if isinstance(old_input_spec, dict):
        for key, sharding_spec in old_input_spec.items():
            # For RowParallelBertOutput, it is assumed the last dimension of input is sharded
            if isinstance(sharding_spec, MeshShardingSpec):
                sharding_spec.dims = tuple([-1])
                sharding_spec.group = group
                sharding_spec.ranks = ranks
                old_input_spec[key] = sharding_spec

    if isinstance(old_output_spec, MeshShardingSpec):
        old_output_spec.dims = tuple()
        old_output_spec.group = group
        old_output_spec.ranks = ranks
    else:
        raise ValueError("Currently only MeshShardingSpec is supported")
    return old_input_spec, old_output_spec


def _megatron_bert_attn_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    """Transform the original input/output spec into the megatron bert form
    Assign the correct group to the specs
    Input/Output are not sharded

    Args:
        old_input_spec is a dict, with keys: input_node name, value: sharding_spec (tuple/list/dict)
        old_output_spec is simply a sharding spec (or a tuple/list/dict of sharding specs)

    Returns:
        input_spec and output_spec for RowParallelLinear module node.
    """

    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


def _megatron_clip_attn_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    """Transform the original input/output spec into the megatron clip form
    Assign the correct group to the specs
    Input/Output are not sharded

    Args:
        old_input_spec is a dict, with keys: input_node name, value: sharding_spec (tuple/list/dict)
        old_output_spec is simply a sharding spec (or a tuple/list/dict of sharding specs)

    Returns:
        input_spec and output_spec for RowParallelLinear module node.
    """
    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)

    if isinstance(output_spec, tuple):
        # clip attention has 2 outputs: attn_output, attn_weights_reshaped
        if old_output_spec[1] is not None:
            # attn_weights_reshaped = attn_weights.view(bsz, self.num_heads_per_partition, tgt_len, src_len)
            # dim 1 of attn_weights_reshaped is sharded
            old_output_spec[1].dims = tuple([1])
            old_output_spec[1].group = group
            old_output_spec[1].ranks = ranks

    return input_spec, output_spec


def _megatron_clip_mlp_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


def _megatron_glm_attn_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    """Transform the original input/output spec into the megatron glm form
    Assign the correct group to the specs
    Input/Output are not sharded

    Args:
        old_input_spec is a dict, with keys: input_node name, value: sharding_spec (tuple/list/dict)
        old_output_spec is simply a sharding spec (or a tuple/list/dict of sharding specs)

    Returns:
        input_spec and output_spec for RowParallelLinear module node.
    """

    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


def _megatron_glm_mlp_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    """Transform the original input/output spec into the megatron glm form
    Assign the correct group to the specs
    Input/Output are not sharded

    Args:
        old_input_spec is a dict, with keys: input_node name, value: sharding_spec (tuple/list/dict)
        old_output_spec is simply a sharding spec (or a tuple/list/dict of sharding specs)

    Returns:
        input_spec and output_spec for RowParallelLinear module node.
    """

    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


def _vocab_parallel_embedding_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    """Input/output are not sharded, just assign groups and ranks"""
    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


def _megatron_glm_block_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


def _megatron_gptneox_attn_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


def _megatron_gptneox_mlp_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


def _megatron_gptneox_layer_spec(old_input_spec, old_output_spec, group="tensor", ranks=None):
    input_spec, output_spec = _empty_spec(old_input_spec, old_output_spec, group, ranks)
    return input_spec, output_spec


# -----------------
# Defining intra node communication
# -----------------


def _row_parallel_linear_com(ranks, tensor_shape, orig_module):
    """Generates all the collective communication operators used by this parallel operator
    and all args needed to compute the cost of each operator

    Args:
        ranks: on which this operator is working on
        tensor_shape: the shape of the output of the original module
        orig_module: the original module

    Return:
        A list of operator names by which we can infer the function to compute the cost.
            Example.
            >>> [("all_reduce", ranks, tensor_shape)]
            Through _COMMUNICATOR_COSTS we compute the cost of all_reduce by calling
            >>> get_all_reduce_cost(ranks, tensor_shape)
        This function should infer the correct shape of tensors distributed, ranks on which communicators are working
        through the input args ranks, tensor_shape, orig_module

    """
    return [("all_reduce", ranks, tensor_shape)]


def _col_parallel_linear_com(ranks, tensor_shape, orig_module):
    return [("all_reduce", ranks, tensor_shape)]


def _vocab_parallel_embedding_com(ranks, tensor_shape, orig_module):
    return [("all_reduce", ranks, tensor_shape)]


def _megatron_bert_attn_com(ranks, tensor_shape, orig_module):
    # 3 all_reduce for 3 ColParallelLinear, 1 all_reduce for 1 RowPrallelLinear
    return [("all_reduce", ranks, tensor_shape)] * 4


def _megatron_glm_attn_com(ranks, tensor_shape, orig_module):
    # 3 all_reduce for 3 ColParallelLinear, 1 all_reduce for 1 RowParallelLinear
    # In fact its 1 all_reduce over a 3*tensor_shape and 1 RowParallelLinear
    return [("all_reduce", ranks, tensor_shape)] * 4


def _megatron_glm_mlp_com(ranks, tensor_shape, orig_module):
    # 1 all_reduce for 1 ColParallelLinear, 1 all_reduce for 1 RowParallelLinear
    return [("all_reduce", ranks, tensor_shape)] * 2


def _megatron_glm_block_com(ranks, tensor_shpae, orig_module):
    return _megatron_glm_attn_com(ranks, tensor_shpae, orig_module.attention) + _megatron_glm_mlp_com(
        ranks, tensor_shpae, orig_module.mlp
    )


def _megatron_clip_attn_com(ranks, tensor_shape, orig_module):
    # 3 all_reduce for 3 ColParallelLinear, 1 all_reduce for 1 RowPrallelLinear
    return [("all_reduce", ranks, tensor_shape)] * 4


def _megatron_clip_mlp_com(ranks, tensor_shape, orig_module):
    # 1 all_reduce for 1 ColParallelLinear, 1 all_reduce for 1 RowParallelLinear
    return [("all_reduce", ranks, tensor_shape)] * 2


def _megatron_gptneox_attn_com(ranks, tensor_shape, orig_module):
    # 3 all_reduce for 3 ColParallelLinear, 1 all_reduce for 1 RowParallelLinear
    # In fact its 1 all_reduce over a 3*tensor_shape and 1 RowParallelLinear
    # for gpt neox, the first output is the attn_output
    return [("all_reduce", ranks, tensor_shape[0])] * 4


def _megatron_gptneox_mlp_com(ranks, tensor_shape, orig_module):
    # 1 all_reduce for 1 ColParallelLinear, 1 all_reduce for 1 RowParallelLinear
    return [("all_reduce", ranks, tensor_shape)] * 2


def _megatron_gptneox_layer_com(ranks, tensor_shpae, orig_module):
    return _megatron_gptneox_attn_com(ranks, tensor_shpae, orig_module.attention) + _megatron_gptneox_mlp_com(
        ranks, tensor_shpae[0], orig_module.mlp
    )


# -----------------
# Defining memory requirements for node
# -----------------


def activation_mem_requirement(ranks, tensor_shape, orig_module):
    """Computes the memory taken up by the tensor of shape tensor_shape"""
    forward_mem = _compute_tensor_size(tensor_shape)

    params_mem = 0
    if isinstance(orig_module, torch.nn.Module):
        for param in orig_module.parameters():
            param_dtype = param.dtype
            param_size_per_elem_bytes = torch.tensor([], dtype=param_dtype).element_size()
            param_size = param.numel() * param_size_per_elem_bytes
            params_mem += param_size

    forward_mem += params_mem
    grad_mem = forward_mem
    return forward_mem, grad_mem, params_mem


def _linear_module_mem_requirement(ranks, tensor_shape, orig_module):
    """Computes the memory requirement of a linear module

    Args:
        ranks: the ranks on which the module is distributed
        tensor_shape: shape of the output tensor of the original module, assuming full replica
        orig_module: the original module instance, this is always the non-parallel module

    Returns:
        the memory requirement in bytes
    """
    weight_dtype = orig_module.weight.dtype
    weight_size_per_elem_bytes = torch.tensor([], dtype=weight_dtype).element_size()
    activation_dtype = tensor_shape.dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    # cost of activation
    activation_size = np.prod(list(tensor_shape.shape)) * activation_size_per_elem_bytes
    # cost of weight and bias
    weight_size = orig_module.weight.numel() * weight_size_per_elem_bytes
    # cost of bias
    if hasattr(orig_module, "bias") and orig_module.bias is not None:
        bias_size = orig_module.bias.numel() * weight_size_per_elem_bytes
    else:
        bias_size = 0
    forward_mem = activation_size
    grad_mem = forward_mem
    params_mem = weight_size + bias_size
    return forward_mem, grad_mem, params_mem


def _row_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module):
    """Computes the memory requirement of a row parallel linear module

    Args:
        ranks: the ranks on which the module is distributed
        tensor_shape: shape of the output tensor of the original module, assuming full replica
        orig_module: the original module instance, this is always the non-parallel module

    Returns:
        the memory requirement in bytes
    """
    weight_dtype = orig_module.weight.dtype
    weight_size_per_elem_bytes = torch.tensor([], dtype=weight_dtype).element_size()
    activation_dtype = tensor_shape.dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    # cost of activation
    activation_size = np.prod(list(tensor_shape.shape)) * activation_size_per_elem_bytes
    # cost of weight and bias
    weight_size = orig_module.weight.numel() * weight_size_per_elem_bytes / len(ranks)
    # cost of bias
    if hasattr(orig_module, "bias") and orig_module.bias is not None:
        bias_size = orig_module.bias.numel() * weight_size_per_elem_bytes
    else:
        bias_size = 0
    forward_mem = activation_size
    grad_mem = forward_mem
    params_mem = weight_size + bias_size
    return forward_mem, grad_mem, params_mem


def _col_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module):
    """Computes the memory requirement of a column parallel linear module

    Args:
        ranks: the ranks on which the module is distributed
        tensor_shape: shape of the output tensor of the original module, assuming full replica
        orig_module: the original module instance, this is always the non-parallel module

    Returns:
        the memory requirement in bytes
    """
    weight_dtype = orig_module.weight.dtype
    weight_size_per_elem_bytes = torch.tensor([], dtype=weight_dtype).element_size()
    activation_dtype = tensor_shape.dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    # cost of activation
    activation_size = np.prod(list(tensor_shape.shape)) * activation_size_per_elem_bytes / len(ranks)
    # cost of weight and bias
    weight_size = orig_module.weight.numel() * weight_size_per_elem_bytes / len(ranks)
    # cost of bias
    if hasattr(orig_module, "bias") and orig_module.bias is not None:
        bias_size = orig_module.bias.numel() * weight_size_per_elem_bytes
    else:
        bias_size = 0
    forward_mem = activation_size
    grad_mem = forward_mem
    params_mem = weight_size + bias_size
    return forward_mem, grad_mem, params_mem


def _embedding_mem_requirement(ranks, tensor_shape, orig_module):
    return _linear_module_mem_requirement(ranks, tensor_shape, orig_module)


def _vocab_parallel_embedding_mem_requirement(ranks, tensor_shape, orig_module):
    return _row_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module)


def _bert_attn_mem_requirement(ranks, tensor_shape, orig_module):
    # we estimates 4 parts: key layer, query layer, value layer, attention
    dense_layer_forward_size, _, dense_params_size = _linear_module_mem_requirement(
        ranks, tensor_shape[0], orig_module.self.query
    )
    batchsize, seq_length, _ = tensor_shape[0].shape
    activation_dtype = tensor_shape[0].dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    atten_size = np.prod([batchsize, seq_length, seq_length]) * activation_size_per_elem_bytes
    forward_mem = 4 * dense_layer_forward_size + atten_size
    grad_mem = forward_mem
    params_mem = 4 * dense_params_size
    return forward_mem, grad_mem, params_mem


def _megatron_bert_attn_mem_requirement(ranks, tensor_shape, orig_module):
    # we estimates 4 parts: key layer, query layer, value layer, attention
    qkv_layer_forward_size, _, qkv_params_size = _col_parallel_linear_mem_requirement(
        ranks, tensor_shape[0], orig_module.self.query
    )
    out_layer_forward_size, _, out_params_size = _row_parallel_linear_mem_requirement(
        ranks, tensor_shape[0], orig_module.output.dense
    )
    batchsize, seq_length, _ = tensor_shape[0].shape
    activation_dtype = tensor_shape[0].dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    atten_size = np.prod([batchsize, seq_length, seq_length]) * activation_size_per_elem_bytes
    forward_mem = 3 * qkv_layer_forward_size + out_layer_forward_size + atten_size
    grad_mem = forward_mem
    params_mem = 3 * qkv_params_size + out_params_size
    return forward_mem, grad_mem, params_mem


def _clip_attn_mem_requirement(ranks, tensor_shape, orig_module):
    # we estimates 4 parts: key layer, query layer, value layer, attention
    dense_layer_forward_size, _, dense_params_size = _linear_module_mem_requirement(
        ranks, tensor_shape[0], orig_module.k_proj
    )
    batchsize, seq_length, _ = tensor_shape[0].shape
    activation_dtype = tensor_shape[0].dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    atten_size = np.prod([batchsize, seq_length, seq_length]) * activation_size_per_elem_bytes
    forward_mem = 4 * dense_layer_forward_size + atten_size
    grad_mem = forward_mem
    params_mem = 4 * dense_params_size
    return forward_mem, grad_mem, params_mem


def _clip_mlp_mem_requirement(ranks, tensor_shape, orig_module):
    fc1_size, _, fc1_params_size = _linear_module_mem_requirement(ranks, tensor_shape, orig_module.fc1)
    fc2_size, _, fc2_params_size = _linear_module_mem_requirement(ranks, tensor_shape, orig_module.fc2)
    forward_mem = fc1_size + fc2_size
    grad_mem = fc1_size + fc2_size
    params_mem = fc1_params_size + fc2_params_size
    return forward_mem, grad_mem, params_mem


def _megatron_clip_mlp_mem_requirement(ranks, tensor_shape, orig_module):
    fc1_size, _, fc1_params_size = _col_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module.fc1)
    fc2_size, _, fc2_params_size = _row_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module.fc1)
    forward_mem = fc1_size + fc2_size
    grad_mem = fc1_size + fc2_size
    params_mem = fc1_params_size + fc2_params_size
    return forward_mem, grad_mem, params_mem


def _megatron_clip_attn_mem_requirement(ranks, tensor_shape, orig_module):
    # we estimates 4 parts: key layer, query layer, value layer, attention
    qkv_layer_forward_size, _, qkv_params_size = _col_parallel_linear_mem_requirement(
        ranks, tensor_shape[0], orig_module.k_proj
    )
    out_layer_forward_size, _, out_params_size = _row_parallel_linear_mem_requirement(
        ranks, tensor_shape[0], orig_module.out_proj
    )
    batchsize, seq_length, _ = tensor_shape[0].shape
    activation_dtype = tensor_shape[0].dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    atten_size = np.prod([batchsize, seq_length, seq_length]) * activation_size_per_elem_bytes
    forward_mem = 3 * qkv_layer_forward_size + out_layer_forward_size + atten_size
    grad_mem = forward_mem
    params_mem = 3 * qkv_params_size + out_params_size
    return forward_mem, grad_mem, params_mem


def _glm_attn_mem_requirement(ranks, tensor_shape, orig_module):
    # we estimates 4 parts: key layer, query layer, value layer, attention
    qkv_shape = tensor_shape
    dense_size, _, dense_params_size = _linear_module_mem_requirement(ranks, qkv_shape, orig_module.dense)
    new_shape = list(qkv_shape.shape)
    new_shape[-1] = new_shape[-1] * 3
    new_shape = torch.Size(new_shape)
    qkv_shape = TensorMetadata(
        shape=new_shape,
        dtype=qkv_shape.dtype,
        requires_grad=qkv_shape.requires_grad,
        stride=qkv_shape.stride,
        memory_format=qkv_shape.memory_format,
        is_quantized=qkv_shape.is_quantized,
        qparams=qkv_shape.qparams,
    )
    qkv_size, _, qkv_params_size = _linear_module_mem_requirement(ranks, qkv_shape, orig_module.query_key_value)
    batchsize, seq_length, _ = tensor_shape.shape
    activation_dtype = tensor_shape.dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    atten_size = np.prod([batchsize, seq_length, seq_length]) * activation_size_per_elem_bytes
    forward_mem = dense_size + qkv_size + atten_size
    grad_mem = forward_mem
    params_mem = dense_params_size + qkv_params_size
    return forward_mem, grad_mem, params_mem


def _glm_mlp_mem_requirement(ranks, tensor_shape, orig_module):
    fc1_size, _, fc1_params_size = _linear_module_mem_requirement(ranks, tensor_shape, orig_module.dense_h_to_4h)
    fc2_size, _, fc2_params_size = _linear_module_mem_requirement(ranks, tensor_shape, orig_module.dense_4h_to_h)
    forward_mem = fc2_size + fc1_size
    grad_mem = fc2_size + fc1_size
    params_mem = fc2_params_size + fc1_params_size
    return forward_mem, grad_mem, params_mem


# ignoring layer norm
def _glm_block_mem_requirement(ranks, tensor_shape, orig_module):
    mlp_forward, mlp_grad, mlp_param = _glm_mlp_mem_requirement(ranks, tensor_shape, orig_module.mlp)
    attention_forward, attention_grad, attention_param = _glm_attn_mem_requirement(
        ranks, tensor_shape, orig_module.attention
    )
    return mlp_forward + attention_forward, mlp_grad + attention_grad, mlp_param + attention_param


def _megatron_glm_attn_mem_requirement(ranks, tensor_shape, orig_module):
    # we estimates 4 parts: key layer, query layer, value layer, attention
    qkv_shape = tensor_shape
    dense_size, _, dense_params_size = _row_parallel_linear_mem_requirement(ranks, qkv_shape, orig_module.dense)
    new_shape = list(qkv_shape.shape)
    new_shape[-1] = new_shape[-1] * 3
    new_shape = torch.Size(new_shape)
    qkv_shape = TensorMetadata(
        shape=new_shape,
        dtype=qkv_shape.dtype,
        requires_grad=qkv_shape.requires_grad,
        stride=qkv_shape.stride,
        memory_format=qkv_shape.memory_format,
        is_quantized=qkv_shape.is_quantized,
        qparams=qkv_shape.qparams,
    )
    qkv_size, _, qkv_params_size = _col_parallel_linear_mem_requirement(ranks, qkv_shape, orig_module.query_key_value)
    batchsize, seq_length, _ = tensor_shape.shape
    activation_dtype = tensor_shape.dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    atten_size = np.prod([batchsize, seq_length, seq_length]) * activation_size_per_elem_bytes
    forward_mem = dense_size + qkv_size + atten_size
    grad_mem = forward_mem
    params_mem = dense_params_size + qkv_params_size
    return forward_mem, grad_mem, params_mem


def _megatron_glm_mlp_mem_requirement(ranks, tensor_shape, orig_module):
    fc1_size, _, fc1_params_size = _col_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module.dense_h_to_4h)
    fc2_size, _, fc2_params_size = _row_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module.dense_4h_to_h)
    forward_mem = fc2_size + fc1_size
    grad_mem = fc2_size + fc1_size
    params_mem = fc2_params_size + fc1_params_size
    return forward_mem, grad_mem, params_mem


# ignoring layer norm
def _megatron_glm_block_mem_requirement(ranks, tensor_shape, orig_module):
    mlp_forward, mlp_grad, mlp_param = _megatron_glm_mlp_mem_requirement(ranks, tensor_shape, orig_module.mlp)
    attention_forward, attention_grad, attention_param = _megatron_glm_attn_mem_requirement(
        ranks, tensor_shape, orig_module.attention
    )
    return mlp_forward + attention_forward, mlp_grad + attention_grad, mlp_param + attention_param


def _gptneox_attn_mem_requirement(ranks, tensor_shape, orig_module):
    # we estimates 4 parts: key layer, query layer, value layer, attention
    # gptneox outputs a tuple
    qkv_shape = tensor_shape[0]
    dense_size, _, dense_params_size = _linear_module_mem_requirement(ranks, qkv_shape, orig_module.dense)
    new_shape = list(qkv_shape.shape)
    new_shape[-1] = new_shape[-1] * 3
    new_shape = torch.Size(new_shape)
    qkv_shape = TensorMetadata(
        shape=new_shape,
        dtype=qkv_shape.dtype,
        requires_grad=qkv_shape.requires_grad,
        stride=qkv_shape.stride,
        memory_format=qkv_shape.memory_format,
        is_quantized=qkv_shape.is_quantized,
        qparams=qkv_shape.qparams,
    )
    qkv_size, _, qkv_params_size = _linear_module_mem_requirement(ranks, qkv_shape, orig_module.query_key_value)
    batchsize, seq_length, _ = qkv_shape.shape
    activation_dtype = qkv_shape.dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    atten_size = np.prod([batchsize, seq_length, seq_length]) * activation_size_per_elem_bytes
    forward_mem = dense_size + qkv_size + atten_size
    grad_mem = forward_mem
    params_mem = dense_params_size + qkv_params_size
    return forward_mem, grad_mem, params_mem


def _gptneox_mlp_mem_requirement(ranks, tensor_shape, orig_module):
    fc1_size, _, fc1_params_size = _linear_module_mem_requirement(ranks, tensor_shape, orig_module.dense_h_to_4h)
    fc2_size, _, fc2_params_size = _linear_module_mem_requirement(ranks, tensor_shape, orig_module.dense_4h_to_h)
    forward_mem = fc2_size + fc1_size
    grad_mem = fc2_size + fc1_size
    params_mem = fc2_params_size + fc1_params_size
    return forward_mem, grad_mem, params_mem


# ignoring layer norm
def _gptneox_layer_mem_requirement(ranks, tensor_shape, orig_module):
    mlp_forward, mlp_grad, mlp_param = _gptneox_mlp_mem_requirement(ranks, tensor_shape[0], orig_module.mlp)
    attention_forward, attention_grad, attention_param = _gptneox_attn_mem_requirement(
        ranks, tensor_shape, orig_module.attention
    )
    return mlp_forward + attention_forward, mlp_grad + attention_grad, mlp_param + attention_param


def _megatron_gptneox_attn_mem_requirement(ranks, tensor_shape, orig_module):
    # we estimates 4 parts: key layer, query layer, value layer, attention
    qkv_shape = tensor_shape[0]
    dense_size, _, dense_params_size = _row_parallel_linear_mem_requirement(ranks, qkv_shape, orig_module.dense)
    new_shape = list(qkv_shape.shape)
    new_shape[-1] = new_shape[-1] * 3
    new_shape = torch.Size(new_shape)
    qkv_shape = TensorMetadata(
        shape=new_shape,
        dtype=qkv_shape.dtype,
        requires_grad=qkv_shape.requires_grad,
        stride=qkv_shape.stride,
        memory_format=qkv_shape.memory_format,
        is_quantized=qkv_shape.is_quantized,
        qparams=qkv_shape.qparams,
    )
    qkv_size, _, qkv_params_size = _col_parallel_linear_mem_requirement(ranks, qkv_shape, orig_module.query_key_value)
    batchsize, seq_length, _ = qkv_shape.shape
    activation_dtype = qkv_shape.dtype
    activation_size_per_elem_bytes = torch.tensor([], dtype=activation_dtype).element_size()
    atten_size = np.prod([batchsize, seq_length, seq_length]) * activation_size_per_elem_bytes
    forward_mem = dense_size + qkv_size + atten_size
    grad_mem = forward_mem
    params_mem = dense_params_size + qkv_params_size
    return forward_mem, grad_mem, params_mem


def _megatron_gptneox_mlp_mem_requirement(ranks, tensor_shape, orig_module):
    fc1_size, _, fc1_params_size = _col_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module.dense_h_to_4h)
    fc2_size, _, fc2_params_size = _row_parallel_linear_mem_requirement(ranks, tensor_shape, orig_module.dense_4h_to_h)
    forward_mem = fc2_size + fc1_size
    grad_mem = fc2_size + fc1_size
    params_mem = fc2_params_size + fc1_params_size
    return forward_mem, grad_mem, params_mem


# ignoring layer norm
def _megatron_gptneox_layer_mem_requirement(ranks, tensor_shape, orig_module):
    mlp_forward, mlp_grad, mlp_param = _megatron_gptneox_mlp_mem_requirement(ranks, tensor_shape[0], orig_module.mlp)
    attention_forward, attention_grad, attention_param = _megatron_gptneox_attn_mem_requirement(
        ranks, tensor_shape, orig_module.attention
    )
    return mlp_forward + attention_forward, mlp_grad + attention_grad, mlp_param + attention_param


# -----------------
# Defining sharded operators' computational costs
# -----------------


def _linear_flops(ranks, tensor_shape, orig_module):
    matrix_params = list(tensor_shape.shape)[:-1] + list(orig_module.weight.size())
    return np.prod(matrix_params)


def _row_parallel_linear_flops(ranks, tensor_shape, orig_module):
    matrix_params = list(tensor_shape.shape)[:-1] + list(orig_module.weight.size())
    return np.prod(matrix_params) / len(ranks)


def _col_parallel_linear_flops(ranks, tensor_shape, orig_module):
    matrix_params = list(tensor_shape.shape)[:-1] + list(orig_module.weight.size())
    return np.prod(matrix_params) / len(ranks)


def _embedding_flops(ranks, tensor_shape, orig_module):
    return _linear_flops(ranks, tensor_shape, orig_module)


def _vocab_parallel_embedding_flops(ranks, tensor_shape, orig_module):
    return _row_parallel_linear_flops(ranks, tensor_shape, orig_module)


def _bert_attn_flops(ranks, tensor_shape, orig_module):
    linear_flops = _linear_flops(ranks, tensor_shape[0], orig_module.self.query)
    batchsize, seq_length, feature_size = tensor_shape[0].shape
    attn_flops = batchsize * seq_length * feature_size * seq_length
    return 4 * linear_flops + attn_flops


def _megatron_bert_attn_flops(ranks, tensor_shape, orig_module):
    qkv_flops = _col_parallel_linear_flops(ranks, tensor_shape[0], orig_module.self.query)
    out_flops = _row_parallel_linear_flops(ranks, tensor_shape[0], orig_module.output.dense)
    batchsize, seq_length, feature_size = tensor_shape[0].shape
    attn_flops = batchsize * seq_length * feature_size * seq_length / len(ranks)
    return qkv_flops * 3 + out_flops + attn_flops


def _clip_mlp_flops(ranks, tensor_shape, orig_module):
    fc2_flops = _linear_flops(ranks, tensor_shape, orig_module.fc2)
    return 2 * fc2_flops


def _megatron_clip_mlp_flops(ranks, tensor_shape, orig_module):
    fc2_flops = _row_parallel_linear_flops(ranks, tensor_shape, orig_module.fc2)
    return 2 * fc2_flops


def _clip_attn_flops(ranks, tensor_shape, orig_module):
    linear_flops = _linear_flops(ranks, tensor_shape[0], orig_module.k_proj)
    batchsize, seq_length, feature_size = tensor_shape[0].shape
    attn_flops = batchsize * seq_length * feature_size * seq_length
    return 4 * linear_flops + attn_flops


def _megatron_clip_attn_flops(ranks, tensor_shape, orig_module):
    qkv_flops = _col_parallel_linear_flops(ranks, tensor_shape[0], orig_module.k_proj)
    out_flops = _row_parallel_linear_flops(ranks, tensor_shape[0], orig_module.out_proj)
    batchsize, seq_length, feature_size = tensor_shape[0].shape
    attn_flops = batchsize * seq_length * feature_size * seq_length / len(ranks)
    return qkv_flops * 3 + out_flops + attn_flops


def _glm_mlp_flops(ranks, tensor_shape, orig_module):
    fc2_flops = _linear_flops(ranks, tensor_shape, orig_module.dense_h_to_4h)
    return 2 * fc2_flops


def _megatron_glm_mlp_flops(ranks, tensor_shape, orig_module):
    fc2_flops = _row_parallel_linear_flops(ranks, tensor_shape, orig_module.dense_4h_to_h)
    return 2 * fc2_flops


def _glm_attn_flops(ranks, tensor_shape, orig_module):
    qkv_shape = tensor_shape
    dense_flops = _linear_flops(ranks, qkv_shape, orig_module.dense)
    new_shape = list(qkv_shape.shape)
    new_shape[-1] = new_shape[-1] * 3
    new_shape = torch.Size(new_shape)
    qkv_shape = TensorMetadata(
        shape=new_shape,
        dtype=qkv_shape.dtype,
        requires_grad=qkv_shape.requires_grad,
        stride=qkv_shape.stride,
        memory_format=qkv_shape.memory_format,
        is_quantized=qkv_shape.is_quantized,
        qparams=qkv_shape.qparams,
    )
    qkv_flops = _linear_flops(ranks, qkv_shape, orig_module.query_key_value)
    batchsize, seq_length, feature_size = tensor_shape.shape
    attn_flops = batchsize * seq_length * feature_size * seq_length
    return dense_flops + qkv_flops + attn_flops


def _megatron_glm_attn_flops(ranks, tensor_shape, orig_module):
    qkv_shape = tensor_shape
    dense_flops = _row_parallel_linear_flops(ranks, qkv_shape, orig_module.dense)
    new_shape = list(qkv_shape.shape)
    new_shape[-1] = new_shape[-1] * 3
    new_shape = torch.Size(new_shape)
    qkv_shape = TensorMetadata(
        shape=new_shape,
        dtype=qkv_shape.dtype,
        requires_grad=qkv_shape.requires_grad,
        stride=qkv_shape.stride,
        memory_format=qkv_shape.memory_format,
        is_quantized=qkv_shape.is_quantized,
        qparams=qkv_shape.qparams,
    )
    qkv_flops = _col_parallel_linear_flops(ranks, qkv_shape, orig_module.query_key_value)
    batchsize, seq_length, feature_size = tensor_shape.shape
    attn_flops = batchsize * seq_length * feature_size * seq_length / len(ranks)
    return dense_flops + qkv_flops + attn_flops


def _glm_block_flops(ranks, tensor_shape, orig_module):
    mlp_flops = _glm_mlp_flops(ranks, tensor_shape, orig_module.mlp)
    attention_flops = _glm_attn_flops(ranks, tensor_shape, orig_module.attention)
    return mlp_flops + attention_flops


def _megatron_glm_block_flops(ranks, tensor_shape, orig_module):
    mlp_flops = _megatron_glm_mlp_flops(ranks, tensor_shape, orig_module.mlp)
    attention_flops = _megatron_glm_attn_flops(ranks, tensor_shape, orig_module.attention)
    return mlp_flops + attention_flops


def _gptneox_mlp_flops(ranks, tensor_shape, orig_module):
    fc2_flops = _linear_flops(ranks, tensor_shape, orig_module.dense_h_to_4h)
    return 2 * fc2_flops


def _megatron_gptneox_mlp_flops(ranks, tensor_shape, orig_module):
    fc2_flops = _row_parallel_linear_flops(ranks, tensor_shape, orig_module.dense_4h_to_h)
    return 2 * fc2_flops


def _gptneox_attn_flops(ranks, tensor_shape, orig_module):
    qkv_shape = tensor_shape[0]
    dense_flops = _linear_flops(ranks, qkv_shape, orig_module.dense)
    new_shape = list(qkv_shape.shape)
    new_shape[-1] = new_shape[-1] * 3
    new_shape = torch.Size(new_shape)
    qkv_shape = TensorMetadata(
        shape=new_shape,
        dtype=qkv_shape.dtype,
        requires_grad=qkv_shape.requires_grad,
        stride=qkv_shape.stride,
        memory_format=qkv_shape.memory_format,
        is_quantized=qkv_shape.is_quantized,
        qparams=qkv_shape.qparams,
    )
    qkv_flops = _linear_flops(ranks, qkv_shape, orig_module.query_key_value)
    batchsize, seq_length, feature_size = tensor_shape[0].shape
    attn_flops = batchsize * seq_length * feature_size * seq_length
    return dense_flops + qkv_flops + attn_flops


def _megatron_gptneox_attn_flops(ranks, tensor_shape, orig_module):
    qkv_shape = tensor_shape[0]
    dense_flops = _row_parallel_linear_flops(ranks, qkv_shape, orig_module.dense)
    new_shape = list(qkv_shape.shape)
    new_shape[-1] = new_shape[-1] * 3
    new_shape = torch.Size(new_shape)
    qkv_shape = TensorMetadata(
        shape=new_shape,
        dtype=qkv_shape.dtype,
        requires_grad=qkv_shape.requires_grad,
        stride=qkv_shape.stride,
        memory_format=qkv_shape.memory_format,
        is_quantized=qkv_shape.is_quantized,
        qparams=qkv_shape.qparams,
    )
    qkv_flops = _col_parallel_linear_flops(ranks, qkv_shape, orig_module.query_key_value)
    batchsize, seq_length, feature_size = tensor_shape[0].shape
    attn_flops = batchsize * seq_length * feature_size * seq_length / len(ranks)
    return dense_flops + qkv_flops + attn_flops


def _gptneox_layer_flops(ranks, tensor_shape, orig_module):
    mlp_flops = _gptneox_mlp_flops(ranks, tensor_shape[0], orig_module.mlp)
    attention_flops = _gptneox_attn_flops(ranks, tensor_shape, orig_module.attention)
    return mlp_flops + attention_flops


def _megatron_gptneox_layer_flops(ranks, tensor_shape, orig_module):
    mlp_flops = _megatron_gptneox_mlp_flops(ranks, tensor_shape[0], orig_module.mlp)
    attention_flops = _megatron_gptneox_attn_flops(ranks, tensor_shape, orig_module.attention)
    return mlp_flops + attention_flops


# -----------------
# Register Sharded Operators
# -----------------

_SHARDABLE_OPERATORS = dict(
    {
        "Linear": torch.nn.Linear,
        "Embedding": torch.nn.Embedding,
        "BertSelfAttention": BertSelfAttention,
        "BertSelfOutput": BertSelfOutput,
        "BertAttention": BertAttention,
        "CLIPAttention": CLIPAttention,
        "CLIPMLP": CLIPMLP,
    }
)

_SHARDED_OPERATORS = dict(
    {
        "RowParallelLinear": RowParallelLinear,
        "ColumnParallelLinear": ColumnParallelLinear,
        "VocabParallelEmbedding": VocabParallelEmbedding,
        "MegatronBertAttention": MegatronBertAttention,
        "RowParallelBertSelfOutput": RowParallelBertSelfOutput,
        "ColumnParallelBertSelfAttention": ColumnParallelBertSelfAttention,
        "MegatronCLIPAttention": MegatronCLIPAttention,
        "MegatronCLIPMLP": MegatronCLIPMLP,
    }
)

_SHARD_MAP = dict(
    {
        "Linear": ["RowParallelLinear", "ColumnParallelLinear"],
        "Embedding": ["VocabParallelEmbedding"],
        "BertSelfAttention": ["ColumnParallelBertSelfAttention"],
        "BertSelfOutput": ["RowParallelBertSelfOutput"],
        "BertAttention": ["MegatronBertAttention"],
        "CLIPAttention": ["MegatronCLIPAttention"],
        "CLIPMLP": ["MegatronCLIPMLP"],
    }
)

_REPLACE_SPECS = dict(
    {
        "RowParallelLinear": _row_linear_spec,
        "ColumnParallelLinear": _col_linear_spec,
        "VocabParallelEmbedding": _vocab_parallel_embedding_spec,
        "ColumnParallelBertSelfAttention": _col_bert_self_attn_spec,
        "RowParallelBertSelfOutput": _row_bert_self_output_spec,
        "MegatronBertAttention": _megatron_bert_attn_spec,
        "MegatronCLIPAttention": _megatron_clip_attn_spec,
        "MegatronCLIPMLP": _megatron_clip_mlp_spec,
    }
)

_ELEMENTWISE_OPS = list(
    [
        torch.nn.Dropout,
        torch.nn.ReLU,
        torch.abs,
        torch.cos,
        torch.exp,
        operator.neg,
        torch.multiply,
        torch.nn.functional.relu,
        torch.nn.functional.dropout,
        torch.flatten,
    ]
)

_RESHAPE_OPS = list([torch.flatten, torch.Tensor.view, torch.reshape])

_BCAST_ELEMENTWISE_OPS = [
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.floor_divide,
    torch.true_divide,
    operator.add,
    operator.sub,
    operator.mul,
    operator.floordiv,
    operator.truediv,
]


_MEMORY_REQUIREMENT = dict(
    {
        "Linear": _linear_module_mem_requirement,
        "RowParallelLinear": _row_parallel_linear_mem_requirement,
        "ColumnParallelLinear": _col_parallel_linear_mem_requirement,
        "Embedding": _embedding_mem_requirement,
        "VocabParallelEmbedding": _vocab_parallel_embedding_mem_requirement,
        "BertAttention": _bert_attn_mem_requirement,
        "MegatronBertAttention": _megatron_bert_attn_mem_requirement,
        "CLIPAttention": _clip_attn_mem_requirement,
        "MegatronCLIPAttention": _megatron_clip_attn_mem_requirement,
        "CLIPMLP": _clip_mlp_mem_requirement,
        "MegatronCLIPMLP": _megatron_clip_mlp_mem_requirement,
    }
)
_INTRA_OPERATOR_COMMUNICATORS = dict(
    {
        "RowParallelLinear": _row_parallel_linear_com,
        "ColumnParallelLinear": _col_parallel_linear_com,
        "VocabParallelEmbedding": _vocab_parallel_embedding_com,
        "MegatronBertAttention": _megatron_bert_attn_com,
        "MegatronCLIPAttention": _megatron_clip_attn_com,
        "MegatronCLIPMLP": _megatron_clip_mlp_com,
    }
)
_OPERATOR_FLOPS = dict(
    {
        "Linear": _linear_flops,
        "RowParallelLinear": _row_parallel_linear_flops,
        "ColumnParallelLinear": _col_parallel_linear_flops,
        "Embedding": _embedding_flops,
        "VocabParallelEmbedding": _vocab_parallel_embedding_flops,
        "BertAttention": _bert_attn_flops,
        "MegatronBertAttention": _megatron_bert_attn_flops,
        "CLIPAttention": _clip_attn_flops,
        "MegatronCLIPAttention": _megatron_clip_attn_flops,
        "CLIPMLP": _clip_mlp_flops,
        "MegatronCLIPMLP": _megatron_clip_mlp_flops,
    }
)


# -----------------
# Register Mock Implementations
# -----------------
def mock_bert_self_attn_layer(
    mod,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=False,
    past_key_value=None,
    output_attentions=False,
):
    shape = hidden_states.shape
    attn_out = torch.empty(shape, device=_DEVICE)
    attn_weight = torch.empty((shape[-2], shape[-2]), device=_DEVICE)
    outputs = (attn_out, attn_weight) if output_attentions else (attn_out,)
    if hasattr(mod, "is_decoder") and mod.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs


register_meta_overrides(BertSelfAttention, mock_bert_self_attn_layer)
register_meta_overrides(BertSelfOutput, mlp_func)
register_meta_overrides(BertAttention, mock_bert_self_attn_layer)


# -----------------
# Resgistry for customized sharded operators
# -----------------


def register_parallel_operator(
    parallel_operator_name,
    parallel_operator_class,
    replace_spec,
    orig_operator_name,
    parallel_mem_req,
    parallel_com_cost,
    orig_operator_class=None,
    mem_req=None,
    orig_operator_flops=None,
    parallel_operator_flops=None,
):
    """Register a distributed operator so that TensorParallelOptimization can use it

    Args:
        parallel_operator_name (str): The unique name given to the distributed operator.
        parallel_operator_class: The actual class of this distributed operator.
        replace_spec (Callable): A callable that transforms the input/output_spec of original operator
            into the distributed input/output_spec of the parallel_operator.
        orig_operator_name (str): The unique name of the original operator.
        parallel_mem_req (Callable): a callable that helps to compute the memory requirement of the sharded operator
        parallel_com_cost (Callable): a callable that helps to compute the communication cost of the sharded operator
        orig_operator_class: The actual class of the original operator. If the orig_operator_class is not registered,
            then this must be provided
        mem_req (Callable): a callable that helps to compute the memory requirement of the original operator
        orig_operator_flops (Callable): a callable that helps to compute the flops of the original operator
        parallel_operator_flops (Callable): a callable that helps to compute the flops of the parallel operator
    """
    if parallel_operator_name in _SHARDED_OPERATORS:
        logger.warning(f"{parallel_operator_name} already exists!")

    if orig_operator_name not in _SHARDABLE_OPERATORS:
        if orig_operator_class is None:
            raise ValueError(
                f"To parallelize {orig_operator_name}, must specify the original operator class by orig_operator_class"
            )
        else:
            _SHARDABLE_OPERATORS[orig_operator_name] = orig_operator_class
            _SHARD_MAP[orig_operator_name] = []
            _MEMORY_REQUIREMENT[orig_operator_name] = mem_req

    _SHARDED_OPERATORS[parallel_operator_name] = parallel_operator_class
    _SHARD_MAP[orig_operator_name].append(parallel_operator_name)
    _REPLACE_SPECS[parallel_operator_name] = replace_spec
    _MEMORY_REQUIREMENT[parallel_operator_name] = parallel_mem_req
    _INTRA_OPERATOR_COMMUNICATORS[parallel_operator_name] = parallel_com_cost
    _OPERATOR_FLOPS[orig_operator_name] = orig_operator_flops
    _OPERATOR_FLOPS[parallel_operator_name] = parallel_operator_flops


def _register_custom_operators():
    try:
        # from transformers_modules.local.modeling_glm import MLP, GLMBlock, SelfAttention
        from antllm.models.hf_glm.modeling_glm import MLP, GLMBlock, SelfAttention
    except ImportError:
        MLP, SelfAttention, GLMBlock = None, None, None
    if MLP:
        register_parallel_operator(
            "MegatronGLMAttention",
            MegatronGLMSelfAttention,
            _megatron_glm_attn_spec,
            "GLMAttention",
            _megatron_glm_attn_mem_requirement,
            _megatron_glm_attn_com,
            orig_operator_class=SelfAttention,
            mem_req=_glm_attn_mem_requirement,
            orig_operator_flops=_glm_attn_flops,
            parallel_operator_flops=_megatron_glm_attn_flops,
        )

        register_meta_overrides(SelfAttention, attention_func)

        register_parallel_operator(
            "MegatronGLMMLP",
            MegatronGLMMLP,
            _megatron_glm_mlp_spec,
            "GLMMLP",
            _megatron_glm_mlp_mem_requirement,
            _megatron_glm_mlp_com,
            orig_operator_class=MLP,
            mem_req=_glm_mlp_mem_requirement,
            orig_operator_flops=_glm_mlp_flops,
            parallel_operator_flops=_megatron_glm_mlp_flops,
        )

        register_meta_overrides(MLP, mlp_func)

        register_parallel_operator(
            "MegatronGLMBlock",
            MegatronGLMBlock,
            _megatron_glm_block_spec,
            "GLMBlock",
            _megatron_glm_block_mem_requirement,
            _megatron_glm_block_com,
            orig_operator_class=GLMBlock,
            mem_req=_glm_block_mem_requirement,
            orig_operator_flops=_glm_block_flops,
            parallel_operator_flops=_megatron_glm_block_flops,
        )

        register_meta_overrides(GLMBlock, transformer_block_func)
    try:
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention, GPTNeoXLayer, GPTNeoXMLP
    except ImportError:
        GPTNeoXAttention, GPTNeoXMLP, GPTNeoXLayer = None, None, None

    if GPTNeoXLayer is not None:

        register_parallel_operator(
            "MegatronGPTNeoXLayer",
            MegatronGPTNeoXLayer,
            _megatron_gptneox_layer_spec,
            "GPTNeoXLayer",
            _megatron_gptneox_layer_mem_requirement,
            _megatron_gptneox_layer_com,
            orig_operator_class=GPTNeoXLayer,
            mem_req=_gptneox_layer_mem_requirement,
            orig_operator_flops=_gptneox_layer_flops,
            parallel_operator_flops=_megatron_gptneox_layer_flops,
        )

        def mock_gpt_neox_layer(
            mod,
            hidden_states,
            attention_mask,
            position_ids,
            head_mask=None,
            layer_past=None,
            use_cache=False,
            output_attentions=False,
        ):
            attn_layer_outputs = mock_gpt_noex_attn(
                hidden_states, attention_mask, position_ids, head_mask, layer_past, use_cache, output_attentions
            )
            attn_output = attn_layer_outputs[0]
            outputs = attn_layer_outputs[1:]
            if use_cache:
                outputs = (attn_output,) + outputs  # hidden_states, present, (attn_weights)
            else:
                outputs = (attn_output,) + outputs[1:]  # hidden_states, (attn_weights)
            return outputs

        def mock_gpt_noex_attn(
            mod,
            hidden_states,
            attention_mask,
            position_ids,
            head_mask=None,
            layer_past=None,
            use_cache=False,
            output_attentions=False,
        ):
            shape = hidden_states.shape

            attn_out = torch.empty(shape, device=_DEVICE)
            attn_weight = torch.empty((shape[-2], shape[-2]), device=_DEVICE)
            present = (attn_out, attn_out) if use_cache else None
            outputs = (attn_weight, present)
            if output_attentions:
                outputs += (attn_weight,)
            return outputs

        register_meta_overrides(GPTNeoXLayer, mock_gpt_neox_layer)

        register_parallel_operator(
            "MegatronGPTNeoXAttention",
            MegatronGPTNeoXAttention,
            _megatron_gptneox_attn_spec,
            "GPTNeoXAttention",
            _megatron_gptneox_attn_mem_requirement,
            _megatron_gptneox_attn_com,
            orig_operator_class=GPTNeoXAttention,
            mem_req=_gptneox_attn_mem_requirement,
            orig_operator_flops=_gptneox_attn_flops,
            parallel_operator_flops=_megatron_gptneox_attn_flops,
        )

        register_meta_overrides(GPTNeoXAttention, mock_gpt_noex_attn)

        register_parallel_operator(
            "MegatronGPTNeoXMLP",
            MegatronGPTNeoXMLP,
            _megatron_gptneox_mlp_spec,
            "GPTNeoXMLP",
            _megatron_gptneox_mlp_mem_requirement,
            _megatron_gptneox_mlp_com,
            orig_operator_class=GPTNeoXMLP,
            mem_req=_gptneox_mlp_mem_requirement,
            orig_operator_flops=_gptneox_mlp_flops,
            parallel_operator_flops=_megatron_gptneox_mlp_flops,
        )
        register_meta_overrides(GPTNeoXMLP, mlp_func)
