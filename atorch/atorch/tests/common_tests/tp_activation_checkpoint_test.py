import os
import tempfile
import unittest

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn import MSELoss
from transformers import set_seed

import atorch
from atorch.distributed.distributed import local_rank, world_size
from atorch.modules.distributed_modules.activation_checkpointing import tp_wrap_fn
from atorch.tests.tp_modules.model_args import ModelArgs


def init_atorch(model_args):
    set_seed(42)

    from atorch.distributed.distributed import create_parallel_group
    from atorch.tests.tp_modules.atorch_mlp import MLP, FeedForward

    tensor_world_size = world_size()

    gpu_partition = ([("tensor", tensor_world_size)], None)
    create_parallel_group(gpu_partition)
    device = f"cuda:{local_rank()}"
    mlp = MLP(model_args).to(device)

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing

    wrap_modules = (FeedForward,)

    def check_fn(module):
        return isinstance(module, wrap_modules)

    apply_activation_checkpointing(
        mlp,
        checkpoint_wrapper_fn=tp_wrap_fn,
        check_fn=check_fn,
    )
    return mlp


def init_fairscale(model_args):
    set_seed(42)
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel

    from atorch.tests.tp_modules.fairscale_mlp import MLP

    world_size = int(os.environ.get("WORLD_SIZE", -1))
    initialize_model_parallel(world_size)
    device = f"cuda:{torch.distributed.get_rank()}"

    mlp = MLP(model_args).to(device)
    return mlp


def run_test(rank, world_size, tmp_file):
    set_seed(42)
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    atorch.init_distributed("nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model_params = ModelArgs(
        dim=64,
        n_layers=4,
        n_heads=8,
        vocab_size=512,
        norm_eps=1e-5,
        max_batch_size=8,
        max_seq_len=8,
    )

    atorch_mlp = init_atorch(model_params)

    fairscale_mlp = init_fairscale(model_params)

    input_ids = torch.randint(
        low=0,
        high=model_params.vocab_size,
        size=(model_params.max_batch_size, model_params.max_seq_len),
        dtype=torch.long,
        device=device,
    )
    labels = torch.rand(model_params.max_batch_size, model_params.max_seq_len, model_params.dim, device=device)

    atorch_loss_fct = MSELoss()
    fairscale_loss_fct = MSELoss()

    atorch_mlp.train()
    fairscale_mlp.train()

    atorch_optimizer = optim.SGD(atorch_mlp.parameters(), lr=0.1)
    fairscale_optimizer = optim.SGD(fairscale_mlp.parameters(), lr=0.1)

    # step: 0, atorch loss: 0.32881444692611694
    # step: 0, fairscale loss: 0.32881444692611694
    # step: 1, atorch loss: 0.373551607131958
    # step: 1, fairscale loss: 0.373551607131958
    for _ in range(2):
        atorch_logits = atorch_mlp(input_ids)
        fairscale_logits = fairscale_mlp(input_ids)
        atorch_optimizer.zero_grad()
        fairscale_optimizer.zero_grad()

        atorch_loss = atorch_loss_fct(atorch_logits.view(-1), labels.view(-1))
        fairscale_loss = fairscale_loss_fct(fairscale_logits.view(-1), labels.view(-1))
        assert torch.allclose(atorch_loss, fairscale_loss), "torch and fairscale should return the same loss"

        atorch_loss.backward(retain_graph=True)
        fairscale_loss.backward(retain_graph=True)
        atorch_optimizer.step()
        fairscale_optimizer.step()

    atorch.reset_distributed()


class TestTPMLP(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkstemp()[1]

    def tearDown(self):
        try:
            os.remove(self.temp)
        except FileNotFoundError:
            pass

    @unittest.skipIf(True, "init_process_group timeout error")
    def test_tp_checkpointing(self):
        world_size = 2
        mp.spawn(
            run_test,
            args=(world_size, self.temp),
            nprocs=world_size,
            join=True,
        )
