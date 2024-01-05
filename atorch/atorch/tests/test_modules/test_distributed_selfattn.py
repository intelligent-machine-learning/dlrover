import math
import os
import sys
import unittest
from argparse import ArgumentParser

import torch
from torch import distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from transformers import set_seed

from atorch.common.util_func import find_free_port
from atorch.modules.distributed_transformer import DistributedSelfAttention

try:
    from transformers.models.bert.modeling_bert import BertConfig, BertSelfAttention
except (ImportError, ModuleNotFoundError):
    from transformers.modeling_bert import BertConfig, BertSelfAttention  # 3.5
global_args = None

SEQ_PARALLEL_PROCESS_GROUP = None


def get_seq_parallel_process_group():
    global SEQ_PARALLEL_PROCESS_GROUP
    if SEQ_PARALLEL_PROCESS_GROUP is None:
        SEQ_PARALLEL_PROCESS_GROUP = _get_default_group()
    return SEQ_PARALLEL_PROCESS_GROUP


def set_seq_parallel_process_group(process_group):
    global SEQ_PARALLEL_PROCESS_GROUP
    SEQ_PARALLEL_PROCESS_GROUP = process_group


class TestFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        t2 = torch.zeros_like(tensor)
        print("TestFunc", tensor.requires_grad, t2.requires_grad)
        return t2

    def backward(ctx, grad):
        return grad


def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {:.1f}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {:.1f}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {:.1f}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {:.1f}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    if torch.distributed.get_rank() == 0:
        # remove subprocess nvidia-smi call because of too slow
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


@unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
class BertSplitTestcase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        set_seed(1234)
        if dist.is_initialized():
            return
        if global_args is None:  # start with python test.py
            torch.distributed.init_process_group(
                "nccl",
                #  init_method="tcp://localhost:6000",
                init_method="tcp://127.0.0.1:12345",
                world_size=1,
                rank=0,
            )
        else:  # start with python -m torch.distributed.launch test.py
            torch.distributed.init_process_group(
                "nccl",
                # init_method="tcp://127.0.0.1:12345",
                rank=global_args.local_rank,
                world_size=int(os.environ["WORLD_SIZE"]),
            )
            torch.cuda.set_device(global_args.local_rank)
        # process_group = torch.distributed.new_group(ranks=[0, 1],
        #                                             backend="nccl")
        # set_seq_parallel_process_group(process_group)

    def tearDown(self) -> None:
        # process_group = get_seq_parallel_process_group()
        # dist.destroy_process_group(process_group)
        return super().tearDown()

    def assertTensorEqual(self, tensor0, tensor1, atol=1e-5):
        self.assertListEqual(list(tensor0.shape), list(tensor1.shape))
        self.assertEqual(tensor0.dtype, tensor1.dtype)
        self.assertFalse(bool(torch.any(torch.isnan(tensor0))))
        self.assertFalse(bool(torch.any(torch.isnan(tensor1))))
        tensor0_str = "mean=%.3f std=%.3f max=%.3f" % (torch.mean(tensor0), torch.std(tensor0), torch.max(tensor0))
        tensor1_str = "mean=%.3f std=%.3f max=%.3f" % (torch.mean(tensor1), torch.std(tensor1), torch.max(tensor1))
        allclose = torch.allclose(tensor0, tensor1, atol=atol, rtol=atol)
        abs_delta = torch.sum(torch.abs(tensor0 - tensor1))
        max_delta = torch.max(tensor0 - tensor1)
        min_delta = torch.min(tensor0 - tensor1)
        self.assertTrue(
            allclose,
            msg=(
                f"tensor not allclose abs={abs_delta:.3f} "
                f"max={max_delta:.4f} min={min_delta:.4f} "
                f"0={tensor0_str} 1={tensor1_str}"
            ),
        )

    def copy_weight(self, src_layer, dst_layer):
        dst_layer.query.weight.data.copy_(src_layer.query.weight.data)
        dst_layer.key.weight.data.copy_(src_layer.key.weight.data)
        dst_layer.value.weight.data.copy_(src_layer.value.weight.data)
        dst_layer.query.bias.data.copy_(src_layer.query.bias.data)
        dst_layer.key.bias.data.copy_(src_layer.key.bias.data)
        dst_layer.value.bias.data.copy_(src_layer.value.bias.data)

    def get_layer_grads(self, layer):
        return [
            (
                "query.weight",
                layer.query.weight.grad,
            ),
            (
                "key.weight",
                layer.key.weight.grad,
            ),
            (
                "value.weight",
                layer.value.weight.grad,
            ),
            (
                "query.bias",
                layer.query.bias.grad,
            ),
            (
                "key.bias",
                layer.key.bias.grad,
            ),
            (
                "value.bias",
                layer.value.bias.grad,
            ),
        ]

    def test_two_layers(self):
        args = [
            [True, True, True, False],
            [True, False, False, False],
            [False, True, False, False],  # some driver version case will hung this testcase
        ]
        for is_async, post_reduce_scatter, profile_distribute, profile_local in args:
            with self.subTest(
                "DistributedSelfAttention",
                is_async=is_async,
                post_reduce_scatter=post_reduce_scatter,
                profile_distribute=profile_distribute,
                profile_local=profile_local,
            ):
                self.__test_two_layers(is_async, post_reduce_scatter, profile_distribute, profile_local)

    def __test_two_layers(self, is_async, post_reduce_scatter, profile_distribute, profile_local):
        # at some cuda version,those values are inconsistent
        # batch_size, hidden_size, seq_len, heads, num_layers, use_fp16 = (
        # 8, 24, 32, 4, 3, False)
        batch_size, hidden_size, seq_len, heads, num_layers, use_fp16 = (50, 768, 1024, 12, 3, False)
        device = torch.cuda.current_device()
        dtype = torch.float32

        hidden_state = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device, requires_grad=False)
        mask = torch.randn(batch_size, 1, 1, seq_len, dtype=dtype, device=device, requires_grad=False)
        process_group = get_seq_parallel_process_group()
        dist.barrier(process_group)
        split_seq_size = seq_len // process_group.size()
        split_hidden_state = torch.split(hidden_state, split_seq_size, dim=1)
        split_mask = torch.split(mask, split_seq_size, dim=3)

        config_dict = dict(
            hidden_size=hidden_size,
            vocab_size_or_config_json_file=119547,
            num_hidden_layers=num_layers,
            num_attention_heads=heads,
            intermediate_size=4 * hidden_size,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            hidden_act="gelu",
            chunk_size_feed_forward=8,
            fp16=use_fp16,
            batch_size=batch_size,
        )
        # without async_op cost:7.196+3.99
        config = BertConfig(**config_dict)
        config.async_op = is_async
        config.post_reduce_scatter = post_reduce_scatter
        layer = DistributedSelfAttention(
            config,
            position_embedding_type="absolute",
            num_micro_q=8,
        )
        atol = 1e-4
        native_layer = BertSelfAttention(config)
        self.copy_weight(layer, native_layer)
        layer.to(device)
        native_layer.to(device)
        this_rank = dist.get_rank(process_group)
        y_slice = [slice(None), slice(this_rank * split_seq_size, (this_rank + 1) * split_seq_size), slice(None)]
        if profile_distribute:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True,
                with_stack=True,
            ) as p:
                for _ in range(10):
                    layer.zero_grad()
                    output = layer(split_hidden_state[this_rank], split_mask[this_rank])
                    loss = torch.mean(torch.pow(torch.rand_like(output[0]) - output[0], 2))
                    loss.backward()

            if this_rank == 0:
                print(p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
            p.export_chrome_trace("distributed_layer%s_%s.json" % (this_rank, is_async))
        if profile_local:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True,
                with_stack=True,
            ) as p:
                native_layer.zero_grad()
                total_output2 = native_layer(hidden_state, mask)[0]
                Y = torch.rand_like(total_output2, requires_grad=True)
                local_loss = torch.mean(torch.pow(Y - total_output2, 2))
                local_loss.backward()
            if this_rank == 0:
                print(p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
        # if not profile_local and not profile_distribute:
        total_output2 = native_layer(hidden_state, mask)[0]
        Y = torch.rand_like(total_output2, requires_grad=True)
        local_loss = torch.mean(torch.pow(Y - total_output2, 2))
        local_loss.backward()
        output2 = total_output2[y_slice]

        local_grads = self.get_layer_grads(native_layer)

        layer.zero_grad()

        output = layer(split_hidden_state[this_rank], split_mask[this_rank])
        self.assertTensorEqual(output[0], output2)

        loss = torch.mean(torch.pow(Y[y_slice] - output[0], 2))
        loss.backward()
        self.assertListEqual(list(output[0].shape), list(output2.shape))
        distribute_grads = self.get_layer_grads(layer)
        for local_grad, dis_grad in zip(local_grads, distribute_grads):
            with self.subTest("grad test", name=local_grad[0], rank=this_rank):
                self.assertTensorEqual(local_grad[1], dis_grad[1], atol=atol)

    def test_matrix(self):
        # single machine ,matrix split
        batch, hidden_size, seq_length = (8, 6, 128)
        # batch, hidden_size, seq_length = (1, 6, 8)
        k = torch.randn(batch, seq_length, hidden_size)
        q = torch.randn(batch, seq_length, hidden_size)
        v = torch.randn(batch, seq_length, hidden_size)
        atol = 1e-7 * q.numel()
        # v = torch.arange(0,batch*seq_length*hidden_size).reshape(batch, seq_length, hidden_size).to(torch.float32)
        attention_mask = torch.randn(batch, 1, seq_length)

        attention_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, seq_length,seq_lenth)
        # head_mask = torch.randn(batch, 1, seq_length)
        attention_scores2 = attention_scores + attention_mask  # boardcast
        # softmax_fn = nn.Softmax(dim=-1)#
        # attention_probs = softmax_fn(attention_scores)
        context = torch.matmul(attention_scores2, v)
        self.assertListEqual(list(attention_scores.shape), [batch, seq_length, seq_length])
        for SPLIT_SIZE in [1, 2, 4]:
            with self.subTest("Splitsize test", SPLIT_SIZE=SPLIT_SIZE):
                sub_seq_length = seq_length // SPLIT_SIZE
                sub_ks = torch.split(k, sub_seq_length, dim=1)
                sub_mask = torch.split(attention_mask, sub_seq_length, dim=2)
                sub_vs = torch.split(v, sub_seq_length, dim=1)
                sub_attn_scores = []
                # Suppose `sub_ks` are currently split across `SPLIT_SIZE` machines
                for ks, sub_m, vs in zip(sub_ks, sub_mask, sub_vs):
                    sub_attn_score = torch.matmul(q, ks.transpose(-1, -2))
                    sub_attn_prob = sub_attn_score + sub_m  # (seq_length, splited_seq)
                    # sub_attn_prob = softmax_fn(sub_attn_prob)# cross node softmax
                    # sub_seq_length*hidden_size
                    sub_attn_prob = torch.matmul(sub_attn_prob, vs)
                    sub_attn_scores.append(sub_attn_prob)
                t = torch.zeros_like(sub_attn_scores[0])
                for attn in sub_attn_scores:  # for multi nodes,there are all-reduce SUM
                    t += attn
                self.assertTrue(torch.allclose(t, context, atol=atol))

    def test_with_transpose(self):
        num_attention_heads, attention_head_size = 3, 2

        def transpose_for_scores(x):
            new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
            x = x.view(*new_x_shape)  # -> (batch,seq_length,num_attention_heads,attention_head_size)
            return x.permute(0, 2, 1, 3)  # -> (batch, num_attention_heads, seq_length,attention_head_size)

        batch, hidden_size, seq_length = (8, num_attention_heads * attention_head_size, 128)
        # batch, hidden_size, seq_length = (1, num_attention_heads * attention_head_size, 4)

        k = torch.randn(batch, seq_length, hidden_size)
        q = torch.randn(batch, seq_length, hidden_size)
        v = torch.randn(batch, seq_length, hidden_size)
        atol = 1e-7 * q.numel()

        attention_mask = torch.randn(batch, 1, 1, seq_length)

        query_layer = transpose_for_scores(q)  # ()
        key_layer = transpose_for_scores(k)
        value_layer = transpose_for_scores(v)
        self.assertListEqual(list(query_layer.shape), [batch, num_attention_heads, seq_length, attention_head_size])
        # -> (batch, num_attention_heads, seq_length,attention_head_size)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention_head_size)
        self.assertListEqual(list(attention_scores.shape), [batch, num_attention_heads, seq_length, seq_length])
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        context = torch.matmul(attention_scores, value_layer)
        for SPLIT_SIZE in [1, 2, 4]:
            sub_seq_length = seq_length // SPLIT_SIZE
            sub_qs = torch.split(query_layer, sub_seq_length, dim=2)
            sub_ks = torch.split(key_layer, sub_seq_length, dim=2)
            sub_mask = torch.split(attention_mask, sub_seq_length, dim=-1)
            sub_vs = torch.split(value_layer, sub_seq_length, dim=2)
            sub_attn_scores = []
            for qs in sub_qs:  # For multi-machine operations, all qs need to be communicated between ranks
                for ks, sub_m in zip(sub_ks, sub_mask):
                    sub_attn_score = torch.matmul(qs, ks.transpose(-1, -2)) / math.sqrt(
                        attention_head_size
                    )  # (splited_seq, splited_seq)
                    # (seq_length, splited_seq) from left to right,from top to bottom
                    sub_attn_prob = sub_attn_score + sub_m
                    sub_attn_scores.append(sub_attn_prob)
            sub_context_ret = [torch.zeros_like(sub_vs[t]) for t in range(SPLIT_SIZE)]  #
            # SPLIT_SIZE*SPLIT_SIZE piece
            for sub_attn_idx, sub_attn_prob in enumerate(sub_attn_scores):
                sub_context_ret[sub_attn_idx // SPLIT_SIZE] = sub_context_ret[
                    sub_attn_idx // SPLIT_SIZE
                ] + torch.matmul(sub_attn_prob, sub_vs[sub_attn_idx % SPLIT_SIZE])
            #
            t = torch.cat(sub_context_ret, dim=2)

            self.assertTensorEqual(context, t, atol)


if __name__ == "__main__":
    """
    1. multi nodes startup: python -m torch.distributed.launch  \
        --nproc_per_node=2  --nnodes=1 --master_addr="127.0.0.1" \
            atorch/tests/test_modules/test_distributed_selfattn.py
    2. single node startup: python atorch/tests/test_modules/test_distributed_selfattn.py
    """
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args, argv = parser.parse_known_args()
    if "WORLD_SIZE" in os.environ:
        # print(args, os.environ["WORLD_SIZE"])
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        # global global_args
        global_args = args
        unittest.main(argv=[sys.argv[0]] + argv)
    else:
        unittest.main()
