import os
import unittest

import pytest

torch = pytest.importorskip("torch", "2.0.9")

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper
except ImportError:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper as ActivationWrapper

import torch.multiprocessing as mp  # noqa: E402
from torch.distributed.fsdp import MixedPrecision  # noqa: E402
from transformers.models.llama.configuration_llama import LlamaConfig  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaDecoderLayer  # noqa: E402

import atorch  # noqa: E402
from atorch.auto.accelerate import auto_accelerate  # noqa: E402
from atorch.common.util_func import find_free_port  # noqa: E402
from atorch.tests.toy_modules.toy_module import get_llama_decoder_layer  # noqa: E402


def boot_test(rank, number_layers, pipes, mode):
    pipe = pipes[rank]
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    torch.manual_seed(1234)
    llama_config = LlamaConfig(use_cache=False)

    class M(torch.nn.Module):
        def __init__(self, llama_config):
            super().__init__()
            self.linears = torch.nn.ModuleList([get_llama_decoder_layer(llama_config) for i in range(number_layers)])

        def forward(self, x):
            for layer in self.linears:
                x = layer(x)[0]
            return x

    dtype = torch.bfloat16
    model = M(llama_config)
    model.cuda()
    fsdp_config = {
        "mixed_precision": MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype),
        "limit_all_gathers": True,
        "use_orig_params": True,
    }
    if mode == "origin":
        checkpoint_config = {
            "wrap_class": LlamaDecoderLayer,
            "no_reentrant": True,
        }
    elif mode == "offload":
        checkpoint_config = {
            "wrap_class": LlamaDecoderLayer,
            "selective_offload": {"offload_args": [("mm.default", [1, 3, 5])], "num_layers": number_layers},
        }
    elif mode == "max_num":
        checkpoint_config = {
            "wrap_class": LlamaDecoderLayer,
            "no_reentrant": True,
            "max_checkpoint_module_num": number_layers - 1,
        }
    strategy = [
        "parallel_mode",
        ("fsdp", fsdp_config),
        ("checkpoint", checkpoint_config),
    ]
    status, result, best_strategy = auto_accelerate(
        model,
        optim_func=torch.optim.AdamW,
        loss_func=lambda x: x,
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        verbose=True,
        ignore_dryrun_on_load_strategy=True,
    )
    assert status
    model = result.model
    opt = result.optim

    checkpointed_num = 0
    for m in model.modules():
        if isinstance(m, ActivationWrapper):
            checkpointed_num += 1

    b = 4
    seq = 2048
    h = 4096
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.MSELoss()
    inp = torch.randn(b, seq, h).cuda().bfloat16() + rank / 10.0
    target = torch.empty(b, seq, h, dtype=torch.bfloat16).fill_(0.0).cuda()
    losses = []
    for i in range(5):
        opt.zero_grad()
        logits = model(inp)
        loss = loss_fn(logits, target)
        loss.backward()
        opt.step()
        losses.append(loss.detach().cpu().item())
    pipe.send((losses, checkpointed_num))


class SelectiveCheckpointTest(unittest.TestCase):
    def setUp(self):
        atorch.reset_distributed()

    def tearDown(self):
        atorch.reset_distributed()

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "Must have at least 2 GPUs for gpu test",
    )
    def test_checkpoint(self):
        """Test origin activation recomputing and offload, compare loss in 5 steps."""
        world_size = 2

        def run_inner(ckpt_mode):
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["NPROC_PER_NODE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(find_free_port())
            rank_pipe = [mp.Pipe() for _ in range(world_size)]
            send, recv = zip(*rank_pipe)
            number_layer = 2
            mp.spawn(
                boot_test,
                args=(number_layer, send, ckpt_mode),
                nprocs=world_size,
                join=True,
                daemon=False,
                start_method="spawn",
            )
            result = []
            for rank, pipe in enumerate(recv):
                result.append(pipe.recv())
            return result

        result = {}
        for ckpt_mode in ["origin", "offload", "max_num"]:
            result[ckpt_mode] = run_inner(ckpt_mode)
        for k in result:
            self.assertListEqual(result["origin"][0][0], result[k][0][0])

        # check max_num's checkpointed_num
        self.assertEqual(result["origin"][0][1], 2)
        self.assertEqual(result["max_num"][0][1], 1)


if __name__ == "__main__":
    unittest.main()
