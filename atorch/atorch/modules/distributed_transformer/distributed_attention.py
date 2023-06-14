import math

import torch
import torch.distributed as dist
from torch import autograd, nn
from torch.distributed import ReduceOp
from torch.utils.checkpoint import checkpoint

from .commu_utils import AllGatherQMicro, ReduceScatterContext

SEQ_STREAMS = None


def get_streams():
    global SEQ_STREAMS
    if SEQ_STREAMS is None:
        SEQ_STREAMS = [torch.cuda.default_stream(), torch.cuda.Stream()]
    return SEQ_STREAMS


class DistributedSoftmax(autograd.Function):
    @staticmethod
    def forward(ctx, s, group=None):
        """
        input:
            s (tensor of [BatchDim, MultiHeadDim, qDim, LocalSeqDim]):
                activation from q*k, aka attention score
            group (ProcessGroup, optional):
                The sequence parallel process group to work on.
        return:
            p (tensor of [BatchDim, MultiHeadDim, qDim, LocalSeqDim]):
                softmaxed attn map, aka attention prob
        """
        # max of max_local, shape: [BatchDim, MultiHeadDim, qDim]
        max_local, _ = s.max(-1)
        max_global = max_local
        dist.all_reduce(max_global, op=ReduceOp.MAX, group=group)

        # sum of sum_local, shape: [BatchDim, MultiHeadDim, qDim]
        s_exp = (s - max_local[..., None]).exp()
        sum_local = s_exp.sum(-1)
        sum_global = sum_local
        dist.all_reduce(sum_global, op=ReduceOp.SUM, group=group)

        p = s_exp / sum_global[..., None]

        # save for backward
        ctx.save_for_backward(p)
        ctx.group = group
        return p

    @staticmethod
    def backward(ctx, p_grad):
        """
        input:
            p_grad([BatchDim, MultiHeadDim, qDim, LocalSeqDim]):
                attention prob's grad
        return:
            s_grad
            (tensor of [BatchDim, MultiHeadDim, qDim, LocalSeqDim]):
                grad of attention score
        """
        (p,) = ctx.saved_tensors

        # distributed softmax backward formula
        # P_i = p_i_grad * p_i, s_i_grad = P_i - p_i_grad * sum(P_j)
        P = p_grad * p
        P_sum_local = P.sum(-1)

        # sum of P_sum_local, shape: [BatchDim, MultiHeadDim, qDim]
        P_sum_global = P_sum_local
        dist.all_reduce(P_sum_global, op=ReduceOp.SUM, group=ctx.group)

        s_grad = P - p * P_sum_global[..., None]
        return s_grad, None


# Based on huggingface's BertSelfAttention
class DistributedSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None, num_micro_q=1, group=None):
        """
        num_micro_q (int): num of step and split micro q
        group: (ProcessGroup, optional): The sequence parallel process group to work on.
        """
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        # note(zy): distributed sequence parallel related attr
        assert self.position_embedding_type == "absolute", f"{self.position_embedding_type}"
        self.num_micro_q = num_micro_q
        self.group = group
        self.async_op = getattr(config, "async_op", False)
        self.stream_size = 2  # compute and communicate overlap
        self.streams = get_streams()
        self.post_reduce_scatter = getattr(config, "post_reduce_scatter", False)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
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
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # note(zy): split q into q_micro
        # [BatchDim, MultiHeadDim, LocalSeqDim, HiddenDim] ->
        # num_micro_q * [BatchDim, MultiHeadDim, microqDim, HiddenDim]
        local_q_micro_list = [t.contiguous() for t in query_layer.chunk(self.num_micro_q, dim=2)]
        c_micro_list = []
        context_layer_list = []
        if self.async_op:
            for stream in self.streams:
                stream.wait_stream(torch.cuda.default_stream())
        micro_q_length = len(local_q_micro_list)
        # TODO(fl): new buffer to store next_query_layer
        next_query_layer = None
        for idx, q_micro in enumerate(local_q_micro_list):
            if self.async_op:
                if idx < micro_q_length - 1:
                    next_micro_q = local_q_micro_list[idx + 1]
                else:
                    next_micro_q = None

                if self.post_reduce_scatter:
                    context_layer = checkpoint(
                        self.softmax_reweight_post, q_micro, key_layer, value_layer, attention_mask, head_mask
                    )
                    context_layer_list.append(context_layer)
                else:
                    if idx:
                        q_micro = next_query_layer
                    next_query_layer, c_micro = checkpoint(
                        self.softmax_reweight_async,
                        q_micro,
                        idx,
                        next_micro_q,
                        key_layer,
                        value_layer,
                        attention_mask,
                        head_mask,
                    )
                    c_micro_list.append(c_micro)
            else:
                c_micro = checkpoint(self.softmax_reweight, q_micro, key_layer, value_layer, attention_mask, head_mask)
                c_micro_list.append(c_micro)
        if self.async_op and self.post_reduce_scatter:
            for context_layer in context_layer_list:
                c_micro = ReduceScatterContext.apply(context_layer, self.group)
                c_micro_list.append(c_micro)
            context_layer_list = []
        if self.async_op:

            default_stream = torch.cuda.default_stream()
            for stream in self.streams:
                default_stream.wait_stream(stream)

        # note(zy): cat back to [BatchDim, MultiHeadDim, LocalSeqDim, HiddenDim]
        context_layer = torch.cat(c_micro_list, dim=2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # note(zy): not support output attentions
        assert not output_attentions
        outputs = (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def softmax_reweight(self, q_micro, key_layer, value_layer, attention_mask, head_mask):
        # num_seq_paral * [BatchDim, MultiHeadDim, microqDim, HiddenDim] ->
        # [BatchDim, MultiHeadDim, qDim, HiddenDim]
        query_layer = AllGatherQMicro.apply(q_micro, self.group)
        context_layer = self.from_qkv_to_context(query_layer, key_layer, value_layer, attention_mask, head_mask)

        # note(zy): reduce scatter back to micro context
        c_micro = ReduceScatterContext.apply(context_layer, self.group)

        return c_micro

    def softmax_reweight_async(self, q_micro, idx, next_q_micro, key_layer, value_layer, attention_mask, head_mask):
        if idx:  # The query_layer prepared by the previous layer
            # because they have have the same stream, so there is no need to wait
            query_layer = q_micro
        else:
            query_layer = AllGatherQMicro.apply(q_micro, self.group)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # Here, test found that AllGather did not have the overlap effect before the from_qkv_to_context
        context_layer = self.from_qkv_to_context(query_layer, key_layer, value_layer, attention_mask, head_mask)
        if next_q_micro is not None:  # let allgather earlier
            with torch.cuda.stream(self.streams[(idx + 1) % self.stream_size]):  # with next stream
                next_query_layer = AllGatherQMicro.apply(next_q_micro, self.group)
        else:
            next_query_layer = None  # reach last
        c_micro = ReduceScatterContext.apply(context_layer, self.group)

        return next_query_layer, c_micro

    def softmax_reweight_post(self, q_micro, key_layer, value_layer, attention_mask, head_mask):
        query_layer = AllGatherQMicro.apply(q_micro, self.group)
        context_layer = self.from_qkv_to_context(query_layer, key_layer, value_layer, attention_mask, head_mask)
        return context_layer

    def from_qkv_to_context(self, query_layer, key_layer, value_layer, attention_mask, head_mask):
        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        seq_length = value_layer.size(2)
        device = value_layer.device
        # note(zy): not support currently
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=device).view(1, -1)
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

        # Normalize the attention scores to probabilities.
        # note(zy): replace with distributed softmax
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = DistributedSoftmax.apply(attention_scores, self.group)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer
