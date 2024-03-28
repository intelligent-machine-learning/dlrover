import os
import unittest

import torch
import torch.multiprocessing as mp

try:
    from transformers import GPTNeoXConfig
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
except ImportError:
    GPTNeoXForCausalLM = None
    GPTNeoXConfig = None


import atorch
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import create_parallel_group, destroy_parallel_group, parallel_group
from atorch.modules.distributed_modules.layers import ColumnParallelLinear
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload, reload_meta_module


class MyModule(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.emb_in = torch.nn.Embedding(vocab_size, hidden_size)
        self.layer0 = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.emb_out = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self._post_init()

    def _post_init(self):
        self.emb_in.weight = self.emb_out.weight

    def forward(self, tokens):
        return self.emb_out(self.layer0(self.emb_in(tokens)))


def init_distributed(rank, world_size):

    if not torch.distributed.is_initialized():
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NPROC_PER_NODE"] = str(world_size)
        if torch.cuda.is_available():
            atorch.init_distributed("nccl")
            torch.cuda.set_device(rank)
        else:
            atorch.init_distributed("gloo")
    if parallel_group("tensor") is None:
        gpu_partition = ([("tensor", world_size)], None)
        create_parallel_group(gpu_partition)


def run_tp_materialize(rank, world_size):
    init_distributed(rank, world_size)
    with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
        model = torch.nn.Linear(64, 64)

    tp_model = ColumnParallelLinear(orig_module=model)
    device = f"cuda:{rank}"
    # make sure tp can reload
    materialize_modules_to_device(tp_model, device=device)


class TestMetaUtils(unittest.TestCase):
    def tearDown(self):
        destroy_parallel_group()
        return super().tearDown()

    @unittest.skipIf(torch.cuda.is_available(), "cpu test")
    def test_init_and_reload(self):
        with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
            model = MyModule(8, 4)

        assert id(model.emb_in.weight) == id(model.emb_out.weight)
        reload_meta_module(model)
        assert id(model.emb_in.weight) == id(model.emb_out.weight)

    @unittest.skipIf(torch.cuda.is_available() or GPTNeoXForCausalLM is None, "cpu test")
    def test_offload_reload(self):
        with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
            config = {
                "hidden_size": 32,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 512,
            }
            config = GPTNeoXConfig(**config)
            model = GPTNeoXForCausalLM(config)

        input_ids = torch.randint(
            low=0,
            high=512,
            size=(16, 24),
            dtype=torch.long,
        )
        labels = torch.rand(16, 24).long()
        input_batch = {"input_ids": input_ids, "labels": labels}
        materialize_modules_to_device(model)
        model(**input_batch)

    @unittest.skipIf(
        (not torch.cuda.is_available() or torch.cuda.device_count() < 2),
        "Must have at least 2 GPUs for tp test",
    )
    def test_tp_reload(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_tp_materialize,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
