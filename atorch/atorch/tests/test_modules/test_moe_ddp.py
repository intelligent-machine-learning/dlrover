import os
import tempfile
import traceback
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from transformers.models.bert import BertConfig
    from transformers.models.bert.modeling_bert import BertLayer, BertModel
except (ImportError, ModuleNotFoundError):
    from transformers.modeling_bert import BertConfig, BertLayer

from atorch.modules.moe import MoEMixtureDistributedDataParallel, set_experts_process_group
from atorch.modules.moe.inject import replace_with_moe
from atorch.utils.version import torch_version


def _run_moe_on_moe_ddp(rank, world_size, experts_size, tmp_file):
    import torch

    try:
        print(torch.cuda.is_available())
        if torch.cuda.is_available():
            device = torch.device("cuda", int(rank))
            torch.cuda.set_device(device)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        # os.environ["MASTER_PORT"] = "12369"
        dist.init_process_group(
            init_method="file://" + tmp_file,
            rank=rank,
            backend=backend,
            world_size=world_size,
        )
        set_experts_process_group(rank, world_size, experts_size)
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
        # TODO: fmoe have compatibility issue:
        # TypeError: ensure_nccl():  incompatible function arguments.
        # The following argument types are supported: arg1: torch.Tensor
        config.moe_impl = "atorch"
        config.moe_gate = "switch"
        model = BertModel(config)
        replace_with_moe(BertLayer, model, config, num_experts=experts_size, moe_impl=config.moe_impl)
        # test all_gather
        MoEDDPTestcase.same_tensor_with_rank(torch.ones(10, 10, device=device), [0, 1, 2, 3])

        device = torch.cuda.current_device()
        model = model.to(device)
        model = MoEMixtureDistributedDataParallel(model)
        input_ids = torch.randint(0, vocab_size_or_config_json_file, (batch_size, seq_len), device=device)
        attention_mask = torch.randint(0, 1, (batch_size, seq_len), device=device)

        model.zero_grad()
        print("before forward")
        output = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        print("after forward")

        Y = torch.randn_like(output)
        loss = torch.sum(torch.pow(Y - output, 2))
        loss.backward()
        print("after backward")
        # TODO: add assert
        # for name, p in model.named_parameters():
        #     if "moe_output.mlp.experts" in name:
        dist.destroy_process_group()
    except Exception:
        traceback.print_exc()
        raise


class MoEDDPTestcase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkstemp()[1]

    def tearDown(self):
        try:
            os.remove(self.temp)
        except FileNotFoundError:
            pass

    @unittest.skipIf(torch_version() >= (2, 0, 0), "torch2.1.0.dev may changed DDP's attribute.")  # type: ignore
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "need 4 gpus for testcase, DP 2 x EP 2",
    )
    def test_moe_init(self):
        world_size = 4
        experts_size = 2

        mp.spawn(
            _run_moe_on_moe_ddp,
            args=(world_size, experts_size, self.temp),
            nprocs=world_size,
            join=True,
        )

    @staticmethod
    def get_tensor_debug_info(tensor, name):
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

    @staticmethod
    def same_tensor_with_rank(tensor, ranks):
        if not isinstance(ranks, list):
            ranks = [ranks]
        recv_tensors = [torch.empty_like(tensor) for i in range(len(ranks))]
        dist.all_gather(recv_tensors, tensor)
        rets = []
        for i in range(len(recv_tensors) - 1):
            rets.append(torch.allclose(recv_tensors[i], recv_tensors[i + 1]))
        return rets


if __name__ == "__main__":
    # launch by unittest?
    unittest.main()
