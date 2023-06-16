"""Distributed Transformers.
If TP is mixed with FSDP, then we defer materialization of the layers until they are wrapped with FSDP.
However, this comes with constraints:
    1. all parameters are transparent to FSDP before wrapping, so we have to construct all parameters,
    even if we move them to meta right after construction
    2. all parameters should be on the same device, so we have to move everything on to meta and wait until
    reset_parameters to load everything back. This means we have to hack around some non shardable layers

FIXME To make TP compatible with module replace optimization, we make the following simplifying assumption:
    1. If TP is mixed with FSDP/Pipeline, always do deferred materialization.
        a. This means if TP is not deferred, we do not need to make sure all params being on the same device
        b This further means we can safely attach the original modules
    2. If deferred materialization is enabled, the original module is always on Meta

If module replace optimization spots a meta module, it will reload the module and reempty it.
So if all parameters are on meta in case of deferred init, we have to make sure all parameters
can be loaded back through reload_meta_module method. Making the previous assumption, init from nonsharded
module will be easy. However, init from empty will be troublesome, will fix this later.
"""
import copy
import math

import torch
from transformers.activations import ACT2FN, gelu

try:
    from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding, apply_rotary_pos_emb
except ImportError:
    RotaryEmbedding, apply_rotary_pos_emb = None, None

from atorch.common.log_utils import default_logger as logger
from atorch.modules.transformer.layers import flash_attn_with_mask_bias
from atorch.utils.meta_model_utils import deepcopy_checkpoint_name, reload_meta_module

from .layers import ATorchTPLayer, ColumnParallelLinear, RowParallelLinear
from .utils import divide, split_tensor_along_shard_dim


class ColumnParallelBertSelfAttention(ATorchTPLayer):
    """Megatron style Parallel bert self-attention layer.
    Exact replication of BertSelfAttention, with Q,K,V column partitioned

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config=dict(),
        orig_module=None,
        process_group="tensor",
        ranks=None,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)

        # FIXME fuse key and query?
        if orig_module is not None:
            self.query = ColumnParallelLinear(
                orig_module=self.orig_module.query,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.key = ColumnParallelLinear(
                orig_module=self.orig_module.key,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.value = ColumnParallelLinear(
                orig_module=self.orig_module.value,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.num_attention_heads = self.orig_module.num_attention_heads
            self.attention_head_size = self.orig_module.attention_head_size

            self.num_attention_heads_per_partition = divide(self.num_attention_heads, self.world_size)
            self.all_head_size = self.orig_module.all_head_size
            self.all_head_size_per_partition = self.num_attention_heads_per_partition * self.attention_head_size

            self.position_embedding_type = self.orig_module.position_embedding_type
            self.dropout = copy.deepcopy(self.orig_module.dropout)
            deepcopy_checkpoint_name(self.dropout, self.orig_module.dropout)
            self.is_decoder = self.orig_module.is_decoder
            if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
                self.max_position_embeddings = orig_module.max_position_embeddings
                self.distance_embedding = copy.deepcopy(self.orig_module.distance_embeddings)
                deepcopy_checkpoint_name(self.distance_embedding, self.orig_module.distance_embedding)

        else:
            raise ValueError("Initialization by args are currently not supported")

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.query._reset_parameters()
        self.key._reset_parameters()
        self.value._reset_parameters()
        reload_meta_module(self.dropout)
        if hasattr(self, "distance_embedding"):
            reload_meta_module(self.distance_embedding)

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        world_size = len(ranks)
        return (
            ColumnParallelLinear.orig_module_shardable(orig_module.query, ranks)
            and ColumnParallelLinear.orig_module_shardable(orig_module.key, ranks)
            and ColumnParallelLinear.orig_module_shardable(orig_module.value, ranks)
            and orig_module.num_attention_heads % world_size == 0
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [s, b, n/p * h/n] -> [s, b, n/p, h/n] -> [s, n/p, b, h/n]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads_per_partition, self.attention_head_size)
        x = x.contiguous().view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        If attention_mask/head_mask/encoder_hidden_states etc are provided, then last dim must
        be partitioned
        """
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = (
                torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).contiguous().view(-1, 1)
            )
            position_ids_r = (
                torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).contiguous().view(1, -1)
            )
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size_per_partition,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class RowParallelBertSelfOutput(ATorchTPLayer):
    """Replication of BertSelfOutput, with row parallel dense
    layer
    """

    def __init__(
        self,
        config=dict(),
        orig_module=None,
        process_group="tensor",
        ranks=None,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if orig_module is not None:
            self.dense = RowParallelLinear(
                orig_module=orig_module.dense,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.LayerNorm = copy.deepcopy(self.orig_module.LayerNorm)
            self.dropout = copy.deepcopy(self.orig_module.dropout)
            deepcopy_checkpoint_name(self.LayerNorm, self.orig_module.LayerNorm)
            deepcopy_checkpoint_name(self.dropout, self.orig_module.dropout)

        else:
            raise ValueError("Initialization by args are currently not supported")

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.dense._reset_parameters()
        reload_meta_module(self.LayerNorm)
        reload_meta_module(self.dropout)

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        return RowParallelLinear.orig_module_shardable(orig_module.dense, ranks)

    def forward(self, hidden_states, input_tensor):
        # hiddent_states are partitioned
        # input_tensor should not be partitioned
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MegatronBertAttention(ATorchTPLayer):
    """Megatron style bert attention layer
    ColumnParallelSelfAttention + RowParallelOutput -> full replica output

    # TODO support prune_heads
    """

    def __init__(
        self,
        config=dict(),
        positition_type=None,
        orig_module=None,
        process_group="tensor",
        ranks=None,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if orig_module is not None:
            self.self = ColumnParallelBertSelfAttention(
                orig_module=orig_module.self,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.output = RowParallelBertSelfOutput(
                orig_module=orig_module.output,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.pruned_heads = set()
        else:
            raise ValueError("Initialization by args are currently not supported")

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.self._reset_parameters()
        self.output._reset_parameters()

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        return ColumnParallelBertSelfAttention.orig_module_shardable(
            orig_module.self, ranks
        ) and RowParallelBertSelfOutput.orig_module_shardable(orig_module.output, ranks)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MegatronCLIPMLP(ATorchTPLayer):
    """Megatron Style MLP for HuggingFace CLIPMLP"""

    def __init__(self, config=None, orig_module=None, process_group="tensor", ranks=None, defer_init=False):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if orig_module is not None:
            self.fc1 = ColumnParallelLinear(
                orig_module=orig_module.fc1,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.activation_fn = orig_module.activation_fn
            self.fc2 = RowParallelLinear(
                orig_module=orig_module.fc2,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
        else:
            raise ValueError("Initialization by args are currently not supported")

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.fc1._reset_parameters()

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        return ColumnParallelLinear.orig_module_shardable(
            orig_module.fc1, ranks
        ) and RowParallelLinear.orig_module_shardable(orig_module.fc2, ranks)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MegatronCLIPAttention(ATorchTPLayer):
    """Megatron Style multi-head attention for HuggingFace CLIPModel"""

    def __init__(self, config=None, orig_module=None, process_group="tensor", ranks=None, defer_init=False):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if orig_module is not None:
            self.k_proj = ColumnParallelLinear(
                orig_module=orig_module.k_proj, process_group=self.process_group, ranks=self.ranks
            )
            self.v_proj = ColumnParallelLinear(
                orig_module=orig_module.v_proj, process_group=self.process_group, ranks=self.ranks
            )
            self.q_proj = ColumnParallelLinear(
                orig_module=orig_module.q_proj, process_group=self.process_group, ranks=self.ranks
            )
            self.out_proj = RowParallelLinear(
                orig_module=orig_module.out_proj, process_group=self.process_group, ranks=self.ranks
            )
            self.pruned_heads = set()
            self.embed_dim = orig_module.embed_dim
            self.head_dim = orig_module.head_dim
            self.num_heads = orig_module.num_heads
            self.num_heads_per_partiton = divide(self.num_heads, self.world_size)
            self.scale = orig_module.scale
            self.dropout = copy.deepcopy(self.orig_module.dropout)
            deepcopy_checkpoint_name(self.dropout, self.orig_module.dropout)
        else:
            raise ValueError("Initialization by args are currently not supported")

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.k_proj._reset_parameters()
        self.v_proj._reset_parameters()
        self.q_proj._reset_parameters()
        self.out_proj._reset_parameters()
        reload_meta_module(self.dropout)

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        world_size = len(ranks)
        return (
            ColumnParallelLinear.orig_module_shardable(orig_module.k_proj, ranks)
            and ColumnParallelLinear.orig_module_shardable(orig_module.v_proj, ranks)
            and ColumnParallelLinear.orig_module_shardable(orig_module.q_proj, ranks)
            and RowParallelLinear.orig_module_shardable(orig_module.out_proj, ranks)
            and orig_module.num_heads % world_size == 0
        )

    def _shape(self, tensor, seq_len, bsz):
        return (
            tensor.contiguous()
            .view(bsz, seq_len, self.num_heads_per_partiton, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=None):
        bsz, tgt_len, embed_dim = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads_per_partiton, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads_per_partiton, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads_per_partiton, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.contiguous().view(bsz, self.num_heads_per_partiton, tgt_len, src_len)
                + causal_attention_mask
            )
            attn_weights = attn_weights.contiguous().view(bsz * self.num_heads_per_partiton, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.contiguous().view(bsz, self.num_heads_per_partiton, tgt_len, src_len) + attention_mask
            )
            attn_weights = attn_weights.contiguous().view(bsz * self.num_heads_per_partiton, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.contiguous().view(bsz, self.num_heads_per_partiton, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.contiguous().view(bsz * self.num_heads_per_partiton, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads_per_partiton, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads_per_partiton, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.contiguous().view(bsz, self.num_heads_per_partiton, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, divide(embed_dim, self.world_size))

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class MegatronGLMSelfAttention(ATorchTPLayer):
    """Megatron Style Self attention for HuggingFace GLM model"""

    def __init__(
        self,
        hidden_size=None,
        num_attention_heads=None,
        attention_dropout_prob=None,
        output_dropout_prob=None,
        init_method=None,
        output_layer_init_method=None,
        attention_scale=1.0,
        orig_module=None,
        process_group="tensor",
        ranks=None,
        use_fa=False,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)
        self.use_fa = use_fa

        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        if orig_module is not None:
            self.query_key_value = ColumnParallelLinear(
                orig_module=orig_module.query_key_value,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.dense = RowParallelLinear(
                orig_module=orig_module.dense,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.hidden_size = orig_module.hidden_size
            self.num_attention_heads = orig_module.num_attention_heads
            self.hidden_size_per_partition = divide(self.hidden_size, self.world_size)
            self.hidden_size_per_attention_head = divide(self.hidden_size, self.num_attention_heads)
            self.num_attention_heads_per_partition = divide(self.num_attention_heads, self.world_size)

            self.attention_scale = orig_module.attention_scale
            self.attention_dropout = copy.deepcopy(self.orig_module.attention_dropout)
            self.output_dropout = copy.deepcopy(self.orig_module.output_dropout)
            deepcopy_checkpoint_name(self.attention_dropout, self.orig_module.attention_dropout)
            deepcopy_checkpoint_name(self.output_dropout, self.orig_module.output_dropout)
        else:

            self.hidden_size_per_partition = divide(hidden_size, self.world_size)
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
            self.num_attention_heads_per_partition = divide(num_attention_heads, self.world_size)
            self.attention_scale = attention_scale
            # Strided linear layer.
            self.query_key_value = ColumnParallelLinear(
                hidden_size, 3 * hidden_size, stride=3, init_method=init_method, defer_init=True
            )
            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

            # Output.
            self.dense = RowParallelLinear(
                hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                defer_init=True,
            )
            self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.query_key_value._reset_parameters()
        self.dense._reset_parameters()
        reload_meta_module(self.attention_dropout)
        reload_meta_module(self.output_dropout)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        world_size = len(ranks)
        return (
            ColumnParallelLinear.orig_module_shardable(orig_module.query_key_value, ranks)
            and orig_module.num_attention_heads % world_size == 0
        )

    def forward(self, hidden_states, ltor_mask, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        query_length = hidden_states.size(1)

        if mem is None:
            mixed_x_layer = self.query_key_value(hidden_states)
            (mixed_query_layer, mixed_key_layer, mixed_value_layer) = split_tensor_along_shard_dim(mixed_x_layer, -1, 3)
        else:
            cat = torch.cat((mem, hidden_states), 1)
            mixed_x_layer = self.query_key_value(cat)
            (mixed_query_layer, mixed_key_layer, mixed_value_layer) = split_tensor_along_shard_dim(mixed_x_layer, -1, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]

        if not self.use_fa:
            # Reshape and transpose [b, np, s, hn]
            query_layer = self._transpose_for_scores(mixed_query_layer)
            key_layer = self._transpose_for_scores(mixed_key_layer)
            value_layer = self._transpose_for_scores(mixed_value_layer)
            if self.attention_scale > 1.0:
                # Raw attention scores. [b, np, s, s]
                attention_scores = torch.matmul(
                    query_layer / math.sqrt(self.attention_scale),
                    key_layer.transpose(-1, -2) / math.sqrt(self.hidden_size_per_attention_head * self.attention_scale),
                )
            else:
                attention_scores = torch.matmul(
                    query_layer, key_layer.transpose(-1, -2) / math.sqrt(self.hidden_size_per_attention_head)
                )

            # Apply the left to right attention mask.
            ltor_mask = ltor_mask.type_as(attention_scores)
            attention_scores = torch.mul(attention_scores, ltor_mask)
            if self.attention_scale > 1.0:
                max_attention_scores = attention_scores.max(dim=-1, keepdim=True)[0]
                attention_scores -= max_attention_scores
                attention_scores *= self.attention_scale
            # if torch.distributed.get_rank() == 0:
            #     print(min_attention_scores, attention_scores.max().item())
            attention_scores = attention_scores + (-65504.0) * (1.0 - ltor_mask)
            # Attention probabilities. [b, np, s, s]
            attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            # with get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

            # Context layer.
            # [b, np, s, hn]
            context_layer = torch.matmul(attention_probs, value_layer)
            # [b, s, np, hn]
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        else:
            # Reshape and transpose [b, s, np, hn]
            query_layer = mixed_query_layer.view(
                *mixed_query_layer.size()[:-1],
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            key_layer = mixed_key_layer.view(
                *mixed_key_layer.size()[:-1],
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            value_layer = mixed_value_layer.view(
                *mixed_value_layer.size()[:-1],
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )

            if ltor_mask.dtype != query_layer.dtype:
                ltor_mask = ltor_mask.to(query_layer.dtype)

            dropout_p = self.attention_dropout.p if self.training else 0.0

            context_layer = flash_attn_with_mask_bias(
                query_layer, key_layer, value_layer, mask=(-65504.0) * (1.0 - ltor_mask), dropout_p=dropout_p
            )

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class MegatronGLMMLP(ATorchTPLayer):
    """Megatron style MLP for glm"""

    def __init__(
        self,
        hidden_size=None,
        output_dropout_prob=None,
        init_method=None,
        output_layer_init_method=None,
        orig_module=None,
        process_group="tensor",
        ranks=None,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if orig_module is None:
            # Set output layer initialization if not provided.
            if output_layer_init_method is None:
                output_layer_init_method = init_method
            # Project to 4h.
            self.dense_h_to_4h = ColumnParallelLinear(
                hidden_size, 4 * hidden_size, init_method=init_method, defer_init=True
            )
            # Project back to h.
            self.dense_4h_to_h = RowParallelLinear(
                4 * hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method,
                defer_init=True,
            )
            self.dropout = torch.nn.Dropout(output_dropout_prob)
        else:
            self.dense_h_to_4h = ColumnParallelLinear(
                orig_module=orig_module.dense_h_to_4h,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.dense_4h_to_h = RowParallelLinear(
                orig_module=orig_module.dense_4h_to_h,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.dropout = copy.deepcopy(self.orig_module.dropout)
            deepcopy_checkpoint_name(self.dropout, self.orig_module.dropout)

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.dense_h_to_4h._reset_parameters()
        self.dense_4h_to_h._reset_parameters()
        reload_meta_module(self.dropout)

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        return ColumnParallelLinear.orig_module_shardable(
            orig_module.dense_h_to_4h, ranks
        ) and RowParallelLinear.orig_module_shardable(orig_module.dense_4h_to_h, ranks)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class MegatronGLMBlock(ATorchTPLayer):
    def __init__(
        self,
        hidden_size=None,
        num_attention_heads=None,
        attention_dropout_prob=None,
        output_dropout_prob=None,
        layernorm_epsilon=None,
        init_method=None,
        output_layer_init_method=None,
        attention_scale=1.0,
        orig_module=None,
        process_group="tensor",
        ranks=None,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        if orig_module is None:
            self.attention = MegatronGLMSelfAttention(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                init_method,
                output_layer_init_method=output_layer_init_method,
                attention_scale=attention_scale,
                process_group=process_group,
                defer_init=True,
            )
            self.hidden_size = hidden_size
            self.layernorm_epsilon = self.layernorm_epsilon
            # Layernorm on the input data.
            self.post_attention_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)
            self.input_layernorm = torch.nn.LayerNorm(self.hidden_size, eps=self.layernorm_epsilon)

            self.mlp = MegatronGLMMLP(
                hidden_size,
                output_dropout_prob,
                init_method,
                output_layer_init_method=output_layer_init_method,
                process_group=process_group,
                defer_init=True,
            )
        else:

            self.attention = MegatronGLMSelfAttention(
                orig_module=orig_module.attention, process_group=process_group, defer_init=True
            )

            self.mlp = MegatronGLMMLP(orig_module=orig_module.mlp, process_group=process_group, defer_init=True)
            self.input_layernorm = copy.deepcopy(self.orig_module.input_layernorm)
            self.post_attention_layernorm = copy.deepcopy(self.orig_module.post_attention_layernorm)
            deepcopy_checkpoint_name(self.input_layernorm, self.orig_module.input_layernorm)
            deepcopy_checkpoint_name(self.post_attention_layernorm, self.orig_module.post_attention_layernorm)

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.attention._reset_parameters()
        self.mlp._reset_parameters()
        reload_meta_module(self.input_layernorm)
        reload_meta_module(self.post_attention_layernorm)

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        return MegatronGLMSelfAttention.orig_module_shardable(
            orig_module.attention, ranks
        ) and MegatronGLMMLP.orig_module_shardable(orig_module.mlp, ranks)

    def forward(self, hidden_states, ltor_mask, mem=None):
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        attention_output = self.attention(layernorm_output, ltor_mask, mem)
        layernorm_input = hidden_states + attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)
        output = layernorm_input + mlp_output

        return output


class MegatronGPTNeoXAttention(ATorchTPLayer):
    def __init__(
        self,
        config=None,
        orig_module=None,
        process_group="tensor",
        ranks=None,
        use_fa=False,
        defer_init=False,
    ):
        super().__init__(orig_module, process_group, ranks, defer_init)
        self.use_fa = use_fa

        if orig_module is None:
            self.num_attention_heads = config.num_attention_heads
            self.hidden_size = config.hidden_size
            self.head_size = divide(self.hidden_size, self.num_attention_heads)

            self.rotary_ndims = int(self.head_size * config.rotary_pct)

            self.max_positions = config.max_position_embeddings
            self.rotary_emb_base = self.rotary_emb_base
            self.register_buffer(
                "bias",
                torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.bool)).view(
                    1, 1, self.max_positions, self.max_positions
                ),
            )

            self.register_buffer("masked_bias", torch.tensor(-1e9))

            self.rotary_emb = RotaryEmbedding(self.rotary_ndims, self.max_positions, base=self.rotary_emb_base)

            self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(
                torch.get_default_dtype()
            )
            self.query_key_value = ColumnParallelLinear(
                config.hidden_size,
                3 * config.hidden_size,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.dense = RowParallelLinear(
                config.hidden_size,
                config.hidden_size,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
        else:
            self.query_key_value = ColumnParallelLinear(
                orig_module=orig_module.query_key_value,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.dense = RowParallelLinear(
                orig_module=orig_module.dense,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )

            self.num_attention_heads = orig_module.num_attention_heads
            self.hidden_size = orig_module.hidden_size
            self.head_size = orig_module.head_size
            self.rotary_ndims = orig_module.rotary_ndims
            self.norm_factor = orig_module.norm_factor
            reload_meta_module(orig_module.rotary_emb)
            if orig_module.bias.is_meta:
                orig_module.bias = torch.load(orig_module.bias.checkpoint_name)

            if orig_module.masked_bias.is_meta:
                orig_module.masked_bias = torch.load(orig_module.masked_bias.checkpoint_name)

            self.register_buffer(
                "bias",
                copy.deepcopy(orig_module.bias),
            )
            self.register_buffer("masked_bias", orig_module.masked_bias)
            # rotary_emb will not be replaced, so we do not care
            self.rotary_emb = copy.deepcopy(self.orig_module.rotary_emb)

            object.__setattr__(self, "rotary_emb_copy", copy.deepcopy(self.rotary_emb))

        object.__setattr__(self, "masked_bias_copy", copy.deepcopy(self.masked_bias))
        object.__setattr__(self, "bias_copy", copy.deepcopy(self.bias))
        self.hidden_size_per_partition = divide(self.hidden_size, self.world_size)
        self.num_attention_heads_per_partition = divide(self.num_attention_heads, self.world_size)
        self.rotary_emb.to("meta")
        self.bias = self.bias.to("meta")
        self.masked_bias = self.masked_bias.to("meta")

        if not self.defer_init:
            self.reset_parameters()

    def _init_emb(self):
        if hasattr(self, "rotary_emb_copy"):
            self.rotary_emb = self.rotary_emb_copy
        else:
            self.rotary_emb = RotaryEmbedding(self.rotary_ndims, self.max_positions, base=self.rotary_emb_base)

        self.register_buffer("bias", self.bias_copy)
        self.register_buffer("masked_bias", self.masked_bias_copy)

    def _reset_parameters(self):
        self.query_key_value._reset_parameters()
        self.dense._reset_parameters()
        self._init_emb()

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        world_size = len(ranks)
        return (
            ColumnParallelLinear.orig_module_shardable(orig_module.query_key_value, ranks)
            and orig_module.num_attention_heads % world_size == 0
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
    ):
        if use_cache or layer_past is not None:
            logger.warning("use_cache/layer_past is not compatible with tp")

        if self.use_fa and output_attentions:
            logger.warning("output_attentions is not compatible with use_fa")
            self.use_fa = False

        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [b, s, h]
        #   --> [b, s, (n/p * 3 * h/n)]
        qkv = self.query_key_value(hidden_states)

        # [b, s, (n/p * 3 * h/n)]
        #   --> [b, s, n/p, 3 * h/n]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads_per_partition, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [b, s, n/p, 3 * h/n] --> 3 [b, n/p, s, h/n]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)  # noqa: E203
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)  # noqa: E203

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]  # noqa: E203
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]  # noqa: E203

        # Compute token offset for rotary embeddings (when decoding)
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]

        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        if not self.use_fa:
            # Cache QKV values
            if has_layer_past:
                past_key = layer_past[0]
                past_value = layer_past[1]
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
            present = (key, value) if use_cache else None

            # Compute attention
            # [b, n/p, s, h/n]
            # cast query and key back
            query = query.to(value.dtype)
            key = key.to(value.dtype)
            attention_mask = attention_mask.to(value.dtype)
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            # Reshape outputs
            # [b, s, h/p]
            attn_output = self._merge_heads(attn_output, self.num_attention_heads_per_partition, self.head_size)
        else:
            # cast query and key back
            query = query.to(value.dtype)
            key = key.to(value.dtype)
            attention_mask = attention_mask.to(value.dtype)
            # [b, n/p, s, h/n]
            attn_output = self._fa(query, key, value, attention_mask, head_mask=head_mask)
            attn_output = attn_output.contiguous()
            attn_output = attn_output.view(
                attn_output.size(0), attn_output.size(1), self.num_attention_heads_per_partition * self.head_size
            )
            # support not the present
            present = None
            attn_weights = None

        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [b, s, h/p]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [b, s, n/p, h/n]
        tensor = tensor.view(new_shape)
        # -> [b, n/p, s, h/n]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [b, n/p, s, h/n]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [b, s, n/p, h/n]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [b, s, h/p]
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]  # noqa: E203

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def _fa(self, query, key, value, attention_mask=None, head_mask=None):
        if head_mask is not None:
            logger.waring("Head mask not supported")
        # 3 * [b, n/p, s, h/n] -> 3 * [n, s, n/p, h/n]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        context_layer = flash_attn_with_mask_bias(
            query,
            key,
            value,
            mask=attention_mask,
            softmax_scale=(
                torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor
            ),
            causal=True,
        )
        return context_layer


class MegatronGPTNeoXMLP(ATorchTPLayer):
    def __init__(self, config=None, orig_module=None, process_group="tensor", ranks=None, defer_init=False):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if orig_module is None:
            self.dense_h_to_4h = ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.dense_4h_to_h = RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.act = ACT2FN[config.hidden_act]
        else:
            self.dense_h_to_4h = ColumnParallelLinear(
                orig_module=orig_module.dense_h_to_4h,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.dense_4h_to_h = RowParallelLinear(
                orig_module=orig_module.dense_4h_to_h,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            # reload_meta_module(self.orig_module.act)
            # self.act = copy.deepcopy(self.orig_module.act)
            self.act = self.orig_module.act

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.dense_4h_to_h._reset_parameters()
        self.dense_h_to_4h._reset_parameters()

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        return ColumnParallelLinear.orig_module_shardable(
            orig_module.dense_h_to_4h, ranks
        ) and RowParallelLinear.orig_module_shardable(orig_module.dense_4h_to_h, ranks)

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class MegatronGPTNeoXLayer(ATorchTPLayer):
    def __init__(self, config=None, orig_module=None, process_group="tensor", ranks=None, defer_init=False):
        super().__init__(orig_module, process_group, ranks, defer_init)

        if orig_module is None:
            self.use_parallel_residual = config.use_parallel_residual
            self.hidden_size = self.hidden_size
            self.layer_norm_eps = config.layer_norm_eps
            self.input_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.attention = MegatronGPTNeoXAttention(config, defer_init=True)
            self.mlp = MegatronGPTNeoXMLP(config, defer_init=True)
        else:
            self.use_parallel_residual = orig_module.use_parallel_residual
            self.attention = MegatronGPTNeoXAttention(
                orig_module=orig_module.attention,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.mlp = MegatronGPTNeoXMLP(
                orig_module=orig_module.mlp,
                process_group=self.process_group,
                ranks=self.ranks,
                defer_init=True,
            )
            self.input_layernorm = copy.deepcopy(self.orig_module.input_layernorm)
            self.post_attention_layernorm = copy.deepcopy(self.orig_module.post_attention_layernorm)
            deepcopy_checkpoint_name(self.input_layernorm, self.orig_module.input_layernorm)
            deepcopy_checkpoint_name(self.post_attention_layernorm, self.orig_module.post_attention_layernorm)

        if not self.defer_init:
            self.reset_parameters()

    def _reset_parameters(self):
        self.attention._reset_parameters()
        self.mlp._reset_parameters()
        reload_meta_module(self.input_layernorm)
        reload_meta_module(self.post_attention_layernorm)

    @staticmethod
    def orig_module_shardable(orig_module, ranks):
        return MegatronGLMSelfAttention.orig_module_shardable(
            orig_module.attention, ranks
        ) and MegatronGLMMLP.orig_module_shardable(orig_module.mlp, ranks)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        if use_cache or layer_past is not None:
            logger.warning("use_cache/layer_past is not compatible with tp")

        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs
