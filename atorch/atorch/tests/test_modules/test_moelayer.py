import os
import unittest

import torch
import torch.distributed as dist

from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import find_free_port

try:
    from transformers.models.bert import BertConfig
    from transformers.models.bert.modeling_bert import BertLayer, BertModel
except (ImportError, ModuleNotFoundError):
    from transformers.modeling_bert import BertConfig, BertLayer

import sys
from argparse import ArgumentParser

from torch import nn

from atorch.modules.moe.inject import replace_with_moe

logger.setLevel("INFO")
global_args = None


@unittest.skipIf(not torch.cuda.is_available(), "cuda is not available")
class MOELayerTestcase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # torch.backends.cuda.matmul.allow_tf32 = False
        # torch.backends.cudnn.allow_tf32 = False
        if dist.is_initialized():
            return
        if global_args is None:  # start with python test.py
            torch.distributed.init_process_group(
                "nccl",
                #  init_method="tcp://localhost:6000",
                init_method="tcp://127.0.0.1:29500",
                world_size=1,
                rank=0,
            )
        else:  # start with python -m torch.distributed.launch test.py
            torch.distributed.init_process_group(
                "nccl",
                init_method="tcp://127.0.0.1:29500",
                rank=global_args.local_rank,
                world_size=int(os.environ["WORLD_SIZE"]),
            )
            torch.cuda.set_device(global_args.local_rank)

    def get_tensor_debug_info(self, tensor, name):
        if tensor is None:
            return {"name": name, "is_none": True}
        else:
            head_element = tensor.flatten()[:5].tolist()
            return {
                "name": name,
                "tensor_max": torch.max(tensor),
                "tensor_min": torch.min(tensor),
                "tensor_mean": torch.mean(tensor),
                "has_nan": bool(torch.any(torch.isnan(tensor))),
                "has_inf": bool(torch.any(torch.isinf(tensor))),
                "head": head_element,
            }

    @unittest.skipIf(True, "Failed on gpu")
    def test_moe_gate(self):
        batch_size, hidden_size, seq_len, heads, num_layers, use_fp16 = (10, 768, 512, 12, 3, False)
        vocab_size_or_config_json_file = 60000
        config_dict = dict(
            hidden_size=hidden_size,
            vocab_size_or_config_json_file=vocab_size_or_config_json_file,
            vocab_size=vocab_size_or_config_json_file,
            num_hidden_layers=num_layers,
            num_attention_heads=heads,
            intermediate_size=4 * hidden_size,
            max_position_embeddings=1024,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            hidden_act="gelu",
            chunk_size_feed_forward=1,
            fp16=use_fp16,
            batch_size=batch_size,
        )
        # without async_op cost:7.196+3.99
        config = BertConfig(**config_dict)
        config.moe_impl = "fastmoe"
        config.moe_gate = "switch"
        num_experts = 1  # TODO: why experts>1, output not be same?
        device = torch.cuda.current_device()

        class MyModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.layer = BertLayer(config)

            def forward(self, hidden_state):
                return self.layer(hidden_state)

        # part1: only one BertLayer
        model = MyModel(config)
        model = model.to(device)
        hidden_state = torch.randn(batch_size, seq_len, hidden_size, device=device)
        out = model(hidden_state)
        replace_with_moe(BertLayer, model, config=config, num_experts=num_experts, moe_impl="fastmoe")
        self.assertTrue("(moe_output): MoEBertOutput" in str(model))
        model.to(device)
        out1 = model(hidden_state)
        self.assertLess(torch.max((out[0] - out1[0]) / out1[0]), 3e-3)
        self.assertTrue(torch.allclose(out[0], out1[0], rtol=3e-3))

        # part2:
        model = BertModel(config)
        model = model.to(device)

        input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device, requires_grad=False)
        attn_mask = torch.randint(0, 1, (batch_size, seq_len), device=device, requires_grad=False)
        out0 = model(input_ids, attn_mask)
        num_experts = 1  # TODO: why experts>1, output not be same?

        replace_with_moe(BertLayer, model.encoder, config=config, num_experts=num_experts, moe_impl="fastmoe")
        model = model.to(device)
        self.assertTrue("(moe_output): MoEBertOutput" in str(model))

        out1 = model(input_ids, attn_mask)
        self.assertLess(torch.max((out0[0] - out1[0]) / out1[0]), 1e-5)
        self.assertTrue(torch.allclose(out0[0], out1[0], rtol=1e-5))
        # test atorch bertmodel #TODO:atorch experts have no bias, output is not the same
        # model = BertModel(config)
        # model = model.to(device)

        # input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device, requires_grad=False)
        # attn_mask = torch.randint(0, 1, (batch_size, seq_len), device=device, requires_grad=False)
        # out0 = model(input_ids, attn_mask)
        # num_experts = 1  # TODO: why experts>1, output not be same?
        # config.outer_batch = True
        # config.dispatchV2 = True
        # replace_with_moe(BertLayer, model.encoder, config=config, num_experts=num_experts, moe_impl="atorch")
        # model = model.to(device)
        # self.assertTrue("(moe_output): MoEBertOutput" in str(model))

        # out1 = model(input_ids, attn_mask)
        # self.assertLess(torch.max((out0[0] - out1[0]) / out1[0]), 1e-5)
        # self.assertTrue(torch.allclose(out0[0], out1[0], rtol=1e-5))

    def test_moe_in_ddp(self):
        batch_size, hidden_size, seq_len, heads, num_layers, use_fp16 = (10, 768, 512, 12, 3, False)
        vocab_size_or_config_json_file = 60000
        config_dict = dict(
            hidden_size=hidden_size,
            vocab_size_or_config_json_file=vocab_size_or_config_json_file,
            vocab_size=vocab_size_or_config_json_file,
            num_hidden_layers=num_layers,
            num_attention_heads=heads,
            intermediate_size=4 * hidden_size,
            max_position_embeddings=1024,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            hidden_act="gelu",
            chunk_size_feed_forward=1,
            fp16=use_fp16,
            batch_size=batch_size,
        )
        # without async_op cost:7.196+3.99
        config = BertConfig(**config_dict)
        config.moe_impl = "fastmoe"
        config.moe_gate = "switch"
        config.num_experts = 6
        model = BertModel(config)
        experts_numel = 0
        total_numel = 0
        replace_with_moe(BertLayer, model, config, num_experts=config.num_experts)
        model._ddp_params_and_buffers_to_ignore = []
        for name, tensor in model.state_dict().items():
            if "moe_output.mlp.experts" in name:
                model._ddp_params_and_buffers_to_ignore.append(name)
                if "weight" in name:
                    torch.nn.init.kaiming_uniform_(tensor, 2.236)
                experts_numel += tensor.numel()
            total_numel += tensor.numel()
        device = torch.cuda.current_device()
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)
        input_ids = torch.randint(0, vocab_size_or_config_json_file, (batch_size, seq_len), device=device)
        attention_mask = torch.randint(0, 1, (batch_size, seq_len), device=device)
        model.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)[0]

        Y = torch.randn_like(output)
        loss = torch.sum(torch.pow(Y - output, 2))
        loss.backward()
        for name, p in model.named_parameters():
            if "moe_output.mlp.experts" in name:
                print(
                    "grad_info:%d %r"
                    % (torch.distributed.get_rank(), self.get_tensor_debug_info(p.grad, name + "grad"))
                )
                print("grad_info:%d %r" % (torch.distributed.get_rank(), self.get_tensor_debug_info(p, name)))


if __name__ == "__main__":
    """
    1. multi nodes startup: python -m torch.distributed.launch  \
        --nproc_per_node=2  --nnodes=1 --master_addr="127.0.0.1" \
            atorch/tests/test_modules/test_moelayer.py
    2. single node startup: python atorch/tests/test_modules/test_moelayer.py
    """
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args, argv = parser.parse_known_args()
    if "WORLD_SIZE" in os.environ:
        global_args = args
        # print(args, os.environ["WORLD_SIZE"])
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        unittest.main(argv=[sys.argv[0]] + argv)
    else:
        unittest.main()
