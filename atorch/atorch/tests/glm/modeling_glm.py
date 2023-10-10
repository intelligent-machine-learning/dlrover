import math

import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, init
from torch.nn.parameter import Parameter
from transformers.activations import gelu
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from atorch.modules.transformer.layers import flash_attn_with_mask_bias


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(mean, std, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = std / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=mean, std=std)

    return init_


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class VocabEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.
    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, config):
        super(VocabEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = config.vocab_size
        self.embedding_dim = config.hidden_size
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None

        self.vocab_start_index = 0
        self.vocab_end_index = self.num_embeddings

        # Allocate weights.
        self.weight = Parameter(torch.Tensor(self.num_embeddings, self.embedding_dim))
        # And initialize.
        init.xavier_normal_(self.weight)

    def forward(self, input_):
        # Get the embeddings.
        output = F.embedding(
            input_, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        return output


class MLP(torch.nn.Module):
    """MLP for GPT2.
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform gelu transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    Arguments:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layer initialization. If None,
                                  use `init_method`.
    """

    def __init__(self, hidden_size, output_dropout_prob, init_method, output_layer_init_method=None):
        super(MLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Project to 4h.
        self.dense_h_to_4h = Linear(hidden_size, 4 * hidden_size)

        # Project back to h.
        self.dense_4h_to_h = Linear(4 * hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states):
        # [b, s, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = gelu(intermediate_parallel)

        # [b, s, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GLMStack(torch.nn.Module):
    """GLM transformer.
    This module takes input from embedding layer and it's output can
    be used directly by a logit layer. It consists of L (num-layers)
    blocks of:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection
    followed by a final layer norm.
    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        checkpoint_activations: if True, checkpoint activations.
        checkpoint_num_layers: number of layers to checkpoint. This
                               is basically the chunk size in checkpoitning.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method_std: standard deviation of the init method which has
                         the form N(0, std).
        use_scaled_init_for_output_weights: If Ture use 1/sqrt(2*num_layers)
                                            scaling for the output weights (
                                            output of self attention and mlp).
    """

    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        max_sequence_length,
        embedding_dropout_prob,
        attention_dropout_prob,
        output_dropout_prob,
        checkpoint_activations,
        checkpoint_num_layers=1,
        layernorm_epsilon=1.0e-5,
        init_method_std=0.02,
        use_scaled_init_for_output_weights=True,
        block_position_encoding=False,
        attention_scale=1.0,
    ):
        super(GLMStack, self).__init__()
        self.hidden_size = hidden_size
        # Store activation checkpoiting flag.
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers

        output_layer_init_method = None
        if use_scaled_init_for_output_weights:
            output_layer_init_method = scaled_init_method(0.0, init_method_std, num_layers)
        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        self.block_position_encoding = block_position_encoding

        # Position embedding (serial).
        if block_position_encoding:
            self.position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
            self.block_position_embeddings = torch.nn.Embedding(max_sequence_length + 1, hidden_size)
            torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
        else:
            self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        # Initialize the position embeddings.
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        def get_layer():

            return GLMBlock(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                unscaled_init_method(init_method_std),
                output_layer_init_method=output_layer_init_method,
                attention_scale=attention_scale,
            )

        # Transformer layers.
        self.layers = torch.nn.ModuleList([get_layer() for _ in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(self, hidden_states, position_ids, attention_mask, memory_states=None):

        batch_size, query_length = hidden_states.size()[:2]
        memory_length = memory_states[0].size(1) if memory_states else 0
        # attention mask is the beginning postion of B region, \in [0, query_len)
        is_scalar = torch.numel(attention_mask) == 1
        is_sep = is_scalar or torch.numel(attention_mask) == batch_size
        if is_sep:
            sep = attention_mask.item() if is_scalar else attention_mask

            # conventional transformer
            def build_mask_matrix(seq_length, sep, memory_length=0):
                m = hidden_states.new_ones((1, seq_length, seq_length))
                m = torch.tril(m)
                if is_scalar:
                    m[0, :, : int(sep)] = 1
                else:
                    m = m.expand(batch_size, -1, -1)
                    ids = torch.arange(seq_length, device=sep.device, dtype=sep.dtype).view(1, -1)
                    mask = ids < sep.view(-1, 1)
                    m = m.masked_fill(mask.unsqueeze(1).expand_as(m), 1)
                if memory_length > 0:
                    m = m.expand(batch_size, -1, -1)
                    m = torch.cat((hidden_states.new_ones((batch_size, seq_length, memory_length)), m), dim=2)
                m = m.unsqueeze(1)
                return m

            attention_mask = build_mask_matrix(query_length, sep, memory_length=memory_length)
        else:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask[:, :, :, -query_length - memory_length :]

        if self.block_position_encoding:
            position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        if self.block_position_encoding:
            block_position_embeddings = self.block_position_embeddings(block_position_ids)
            hidden_states = hidden_states + block_position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        def check_detach(_hidden_states):
            return _hidden_states.detach()

        mem_layers = [check_detach(hidden_states)]

        for i, layer in enumerate(self.layers):

            args = [hidden_states, attention_mask]

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs)

                return custom_forward

            mem_i = memory_states[i] if memory_states else None

            if self.checkpoint_activations:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    mem=mem_i,
                )
            else:
                hidden_states = layer(*args, mem=mem_i)
            mem_layers.append(check_detach(hidden_states))

        # Final layer norm.
        output = self.final_layernorm(hidden_states)
        mem_layers = self.update_mems(mem_layers, memory_states)
        return (output, mem_layers)

    def update_mems(self, hiddens, mems):
        memory_length = mems[0].size(1) if mems else 0
        query_length = hiddens[0].size(1)
        new_memory_length = memory_length + query_length

        new_mems = []
        # with torch.no_grad():
        for i in range(len(hiddens)):
            if new_memory_length <= query_length:
                new_mems.append(hiddens[i][:, -new_memory_length:])
            else:
                new_mems.append(torch.cat((mems[i][:, -new_memory_length + query_length :], hiddens[i]), dim=1))
        return new_mems


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.float, learnable=False):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer("inv_freq", inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


@torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), F.embedding(
        position_id, sin.squeeze(1)
    ).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


class SelfAttention(torch.nn.Module):
    """self-attention layer for GLM.
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

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        init_method,
        output_layer_init_method=None,
        attention_scale=1.0,
        use_rotary=False,
        use_fa=False,
    ):
        self.use_rotary = use_rotary
        self.use_fa = use_fa
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        # Per attention head and per partition values.
        self.hidden_size = hidden_size
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)

        self.num_attention_heads = num_attention_heads
        self.attention_scale = attention_scale
        # Strided linear layer.
        self.query_key_value = Linear(hidden_size, 3 * hidden_size)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = Linear(hidden_size, hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        if self.use_rotary:
            self.position_encoding_2d = True
            self.rotary_emb = RotaryEmbedding(
                self.hidden_size // (self.num_attention_heads * 2),
                base=10000,
                precision=torch.half,
                learnable=False,
            )

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ltor_mask, position_ids=None, mem=None):
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

        if not self.use_fa:
            # Reshape and transpose [b, np, s, hn]
            query_layer = self._transpose_for_scores(mixed_query_layer)
            key_layer = self._transpose_for_scores(mixed_key_layer)
            value_layer = self._transpose_for_scores(mixed_value_layer)

            if self.use_rotary:
                query_layer = query_layer.permute(0, 2, 1, 3)
                key_layer = key_layer.permute(0, 2, 1, 3)
                q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
                k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
                cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
                position_ids, block_position_ids = (
                    position_ids[:, 0, :].contiguous(),
                    position_ids[:, 1, :].contiguous(),
                )
                # print(f'q1: {q1.dtype}, k1: {k1.dtype}, q2: {q2.dtype}, k2: {k2.dtype}')
                q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
                q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
                # print(f'qq1: {q1.dtype}, kk1: {k1.dtype}, qq2: {q2.dtype}, kk2: {k2.dtype}')
                query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
                key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))
                query_layer = query_layer.permute(0, 2, 1, 3)
                key_layer = key_layer.permute(0, 2, 1, 3)
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
            query_layer = mixed_query_layer.view(
                *mixed_query_layer.size()[:-1], self.num_attention_heads, self.hidden_size_per_attention_head
            )
            key_layer = mixed_key_layer.view(
                *mixed_key_layer.size()[:-1], self.num_attention_heads, self.hidden_size_per_attention_head
            )
            value_layer = mixed_value_layer.view(
                *mixed_value_layer.size()[:-1], self.num_attention_heads, self.hidden_size_per_attention_head
            )

            if self.use_rotary:

                q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
                k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
                cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
                position_ids, block_position_ids = (
                    position_ids[:, 0, :].contiguous(),
                    position_ids[:, 1, :].contiguous(),
                )
                # print(f'q1: {q1.dtype}, k1: {k1.dtype}, q2: {q2.dtype}, k2: {k2.dtype}')
                q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
                q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
                # print(f'qq1: {q1.dtype}, kk1: {k1.dtype}, qq2: {q2.dtype}, kk2: {k2.dtype}')
                query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1)).half()  # todo fix output float32
                key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1)).half()

            if ltor_mask.dtype != query_layer.dtype:
                ltor_mask = ltor_mask.to(query_layer.dtype)

            dropout_p = self.attention_dropout.p if self.training else 0.0

            context_layer = flash_attn_with_mask_bias(
                query_layer.half(),
                key_layer.half(),
                value_layer.half(),
                mask=(-65504.0) * (1.0 - ltor_mask),
                dropout_p=dropout_p,
            )
            context_layer = context_layer.float()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class GLMBlock(torch.nn.Module):
    """A single layer transformer for GLM.
    We use the following notation:
        h: hidden size
        n: number of attention heads
        b: batch size
        s: sequence length
    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self
                             attention.
        attention_dropout_prob: dropout probability of the attention
                                score in self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid
                           division by zero.
        init_method: initialization method used for the weights. Note
                     that all biases are initialized to zero and
                     layernorm weight are initialized to one.
        output_layer_init_method: output layers (attention output and
                                  mlp output) initialization. If None,
                                  use `init_method`.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        layernorm_epsilon,
        init_method,
        output_layer_init_method=None,
        attention_scale=1.0,
    ):
        super(GLMBlock, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            attention_scale=attention_scale,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = MLP(hidden_size, output_dropout_prob, init_method, output_layer_init_method=output_layer_init_method)

    def forward(self, hidden_states, ltor_mask, mem=None):
        # hidden_states: [b, s, h]
        # ltor_mask: [b,1, s,s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        mem = self.input_layernorm(mem) if mem is not None else None
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask, mem)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


logger = logging.get_logger(__name__)


class GLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~GLMModel`].
    It is used to instantiate an GLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the GLM [shunxing1234/GLM-base-cased](https://huggingface.co/shunxing1234/GLM-base-cased) architecture.
    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the GLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~GLMModel`] or
            [`~TFGLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`~GLMModel`] or
            [`~TFGLMModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        Example:
    ```python
    >>> from transformers import GLMModel, GLMConfig
    >>> # Initializing a GLM shunxing1234/GLM-base-cased style configuration
    >>> configuration = GLMConfig()
    >>> # Initializing a model from the shunxing1234/GLM-base-cased style configuration
    >>> model = GLMModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "glm"
    attribute_map = {"num_hidden_layers": "num_layers"}

    def __init__(
        self,
        num_layers=24,
        vocab_size=30592,
        hidden_size=1024,
        num_attention_heads=16,
        embedding_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        max_sequence_length=512,
        checkpoint_activations=False,
        checkpoint_num_layers=1,
        parallel_output=True,
        relative_encoding=False,
        block_position_encoding=True,
        output_predict=False,
        spell_length=None,
        spell_func="lstm",
        attention_scale=1.0,
        initializer_range=0.02,
        pool_token="cls",
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        self.output_predict = output_predict
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale
        self.initializer_range = initializer_range
        self.pool_token = pool_token
        super().__init__(**kwargs)


class GLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = GLMConfig
    base_model_prefix = "glm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, torch.nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMModel):
            module.gradient_checkpointing = value


class GLMModel(GLMPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`.
    To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an
    `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.output_predict = config.output_predict
        # Word embeddings (parallel).
        self.word_embeddings = VocabEmbedding(config)

        # Transformer
        self.transformer = GLMStack(
            config.num_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.max_sequence_length,
            config.embedding_dropout_prob,
            config.attention_dropout_prob,
            config.output_dropout_prob,
            config.checkpoint_activations,
            config.checkpoint_num_layers,
            attention_scale=config.attention_scale,
            block_position_encoding=config.block_position_encoding,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids=None, position_ids=None, attention_mask=None, mems=None, **kwargs):
        batch_size = input_ids.size(0)
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings

        device = input_ids.device
        input_shape = input_ids.size()

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            block_position_ids = torch.zeros(input_shape[-1], dtype=torch.long, device=device)
            position_ids = torch.stack((position_ids, block_position_ids), dim=0).unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.zeros(batch_size)
        # Transformer.
        transformer_output = self.transformer(embeddings, position_ids, attention_mask, mems)
        last_hidden_states, mems = transformer_output
        logits = None
        if self.output_predict:
            logits = F.linear(last_hidden_states, self.word_embeddings.weight)

        return ModelOutput(
            last_hidden_states=last_hidden_states,
            logits=logits,
            mems=mems,
        )


if __name__ == "__main__":
    config = GLMConfig()
    print(config)
    glm_model = GLMModel(config)
    print(glm_model)

    class FakeInput:
        input_ids = torch.ones((4, 10), dtype=torch.long)
        attention_mask = torch.ones((4, 10))

    res = glm_model(input_ids=FakeInput.input_ids, attention_mask=FakeInput.attention_mask)
    print(res)
