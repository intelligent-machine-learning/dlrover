#!/usr/bin/env python
# coding=utf-8
import copy
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _get_default_group
from transformers.activations import gelu

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import _DistributedContext, create_parallel_group
from atorch.modules.distributed_modules.layers import ColumnParallelLinear, RowParallelLinear, _initialize_affine_weight
from atorch.modules.distributed_modules.transformer import MegatronGLMMLP

logger.setLevel("INFO")
os.environ["NCCL_DEBUG"] = "ERROR"


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    if torch.cuda.is_available():
        atorch.init_distributed("nccl")
    else:
        atorch.init_distributed("gloo")


def _run_row_paralle_linear(rank, world_size):
    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")

    linear = torch.nn.Linear(4, 2).to(device)
    empty_weight = torch.empty(size=linear.weight.shape, device=device)
    empty_weight.copy_(linear.weight)

    ranks = [i for i in range(world_size)]
    row_parallel_linear = RowParallelLinear(orig_module=linear, process_group=pg, ranks=ranks, defer_init=False).to(
        device
    )

    if rank == 0:
        assert torch.norm(empty_weight[:, :2] - row_parallel_linear.weight, p=-1) == 0
    elif rank == 1:
        assert torch.norm(empty_weight[:, 2:] - row_parallel_linear.weight, p=-1) == 0
    atorch.reset_distributed()


def _run_column_paralle_linear(rank, world_size):
    init_dist(rank, world_size)

    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")

    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg, ranks = atorch.distributed.distributed.parallel_group_and_ranks("tensor")

    # init_dist(rank, world_size)
    torch.manual_seed(1)
    linear = torch.nn.Linear(4, 2).to(device)
    linear_copy = copy.deepcopy(linear)
    empty_weight = torch.empty(size=linear.weight.shape, device=device)
    empty_weight.copy_(linear.weight)

    column_parallel_linear = ColumnParallelLinear(
        orig_module=linear, process_group=pg, ranks=ranks, defer_init=False
    ).to(device)

    if rank == 0:
        assert torch.norm(empty_weight[:1, :] - column_parallel_linear.weight, p=2) == 0
    elif rank == 1:
        assert torch.norm(empty_weight[1:, :] - column_parallel_linear.weight, p=2) == 0

    inputs = torch.ones(1, 4).to(device)
    out1 = linear_copy(inputs)
    out2 = column_parallel_linear(inputs)
    print(out1, out2)
    atorch.reset_distributed()


def _run_megatron_mlp(rank, world_size):
    init_dist(rank, world_size)
    config = [("tensor", world_size)]
    create_parallel_group((config, None))
    pg = _get_default_group()
    ranks = [i for i in range(world_size)]
    torch.manual_seed(1)

    class GLMMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense_h_to_4h = torch.nn.Linear(4, 2)
            self.dense_4h_to_h = torch.nn.Linear(2, 4)
            self.dropout = torch.nn.Dropout(1)

        def forward(self, x):
            x = self.dense_h_to_4h(x)
            x = gelu(x)
            x = self.dense_4h_to_h(x)
            x = self.dropout(x)
            return x

    if torch.cuda.is_available():
        device = torch.device(atorch.local_rank())
    else:
        device = torch.device("cpu")
    mlp = GLMMLP().to(device)
    parameter_size = [p.numel() for p in mlp.parameters()]
    parameter_shape = [p.size() for p in mlp.parameters()]
    all_params = [torch.flatten(p) for p in mlp.parameters()]
    all_params = torch.cat(all_params)
    dist.broadcast(all_params, src=0)
    # broadcast
    acc = 0
    for shape, num, p in zip(parameter_shape, parameter_size, mlp.parameters()):
        start = acc
        end = acc + num
        p.data = all_params[start:end].view(shape)

    copy_mlp = copy.deepcopy(mlp)

    mgatron_glm_mlp = MegatronGLMMLP(
        output_dropout_prob=1, orig_module=mlp, ranks=ranks, process_group=pg, defer_init=False
    ).to(device)

    if rank == 0:
        assert torch.norm(copy_mlp.dense_h_to_4h.weight[:1, :] - mgatron_glm_mlp.dense_h_to_4h.weight, p=-1) == 0
        assert torch.norm(copy_mlp.dense_4h_to_h.weight[:, :1] - mgatron_glm_mlp.dense_4h_to_h.weight, p=-1) == 0

    copy_mlp.eval()
    mgatron_glm_mlp.eval()

    input_x = torch.tensor([[1.0, 2.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0], [1.0, 2.0, 2.0, 2.0]]).to(
        device
    )

    y_mlp = copy_mlp(input_x)
    y_megatron_glm_mlp = mgatron_glm_mlp(input_x)

    res = torch.norm(y_mlp - y_megatron_glm_mlp, p=1)
    atorch.reset_distributed()
    assert res == 0


class TestInitializeAffineWeight(unittest.TestCase):
    def test_initialize_affine_weight(self):
        name = "tensor"
        _DistributedContext.PARALLEL_RANK = {}
        _DistributedContext.PARALLEL_GROUP_SIZE = {}
        _DistributedContext.PARALLEL_RANK["tensor"] = 0
        _DistributedContext.PARALLEL_GROUP_SIZE["tensor"] = 2
        master_weight = torch.tensor([1, 2, 3, 4])
        weight = torch.empty(2)
        _initialize_affine_weight(
            weight, per_partition_size=2, master_weight=master_weight, partition_dim=-1, group_name=name
        )
        _DistributedContext.PARALLEL_RANK = None
        _DistributedContext.PARALLEL_GROUP_SIZE = None
        assert torch.norm(weight - master_weight[0:2], p=-1) == 0


class TestRowParallelLinear(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_row_paralle_linear(self):

        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5000"
        mp.spawn(
            _run_row_paralle_linear,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


class TestColumnParallelLinear(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with gpu_num >=2")
    def test_column_paralle_linear(self):

        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5000"
        mp.spawn(
            _run_column_paralle_linear,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


class TestMegatronMLP(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "run with cpu or gpu_num >=2")
    def test_megatron_mlp(self):

        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = "5000"
        mp.spawn(
            _run_megatron_mlp,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
