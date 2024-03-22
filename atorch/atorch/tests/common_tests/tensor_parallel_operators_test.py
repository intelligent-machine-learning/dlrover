"""This test is used to verify the basic tp operators has the expected forward/backward behavior
"""
import os
import unittest

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn import MSELoss
from transformers import set_seed

import atorch
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import destroy_parallel_group, local_rank, world_size
from atorch.tests.tp_modules.model_args import ModelArgs


def init_atorch(model_args):
    set_seed(42)

    from atorch.distributed.distributed import create_parallel_group
    from atorch.tests.tp_modules.atorch_mlp import MLP

    tensor_world_size = world_size()

    gpu_partition = ([("tensor", tensor_world_size)], None)
    create_parallel_group(gpu_partition)
    device = f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu"
    mlp = MLP(model_args).to(device)
    return mlp


def init_fairscale(model_args):
    set_seed(42)
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel

    from atorch.tests.tp_modules.fairscale_mlp import MLP

    world_size = int(os.environ.get("WORLD_SIZE", -1))
    initialize_model_parallel(world_size)

    if torch.cuda.is_available():
        device = f"cuda:{torch.distributed.get_rank()}"
    else:
        device = "cpu"

    mlp = MLP(model_args).to(device)
    return mlp


def run_test(rank, world_size):
    set_seed(42)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    if torch.cuda.is_available():
        atorch.init_distributed("nccl")
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        atorch.init_distributed("gloo")
        device = "cpu"

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
    def tearDown(self):
        destroy_parallel_group()
        return super().tearDown()

    @unittest.skipIf(torch.cuda.is_available() and torch.cuda.device_count() != 2, "run with cpu or gpu_num >=2")
    def test_tp_mlp(self):
        world_size = torch.cuda.device_count()
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_test,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""
