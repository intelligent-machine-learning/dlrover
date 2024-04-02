import functools
import itertools
import math
import os
import unittest
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from atorch.auto.accelerate import auto_accelerate
from atorch.auto.clip_grad_norm import clip_grad_norm as auto_acc_clip_grad_norm
from atorch.auto.clip_grad_norm import fsdp_grad_norm_per_param, fsdp_param_norm_and_grad_num_zero
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import init_distributed, parallel_group, reset_distributed
from atorch.tests.toy_modules.toy_module import ToyCustomModule, ToyModel, loss_func
from atorch.utils.version import torch_version


def _fsdp_strategy_clip_grad_norm(rank, world_size, free_port):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    local_model = ToyModel(use_custom_module=True)
    atorch_wrap_cls = {
        ToyCustomModule,
    }
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=atorch_wrap_cls)
    fsdp_model = FSDP(
        deepcopy(local_model).to(device), sharding_strategy=ShardingStrategy.FULL_SHARD, auto_wrap_policy=wrap_policy
    )
    fsdp_strategy = [
        ("parallel_mode", ([("data", torch.distributed.get_world_size())], None)),
        ("fsdp", {"atorch_wrap_cls": atorch_wrap_cls}),
    ]
    status, result, _ = auto_accelerate(
        model=local_model,
        load_strategy=fsdp_strategy,
        ignore_dryrun_on_load_strategy=True,
    )
    assert status is True
    auto_acc_model = result.model

    for model in (fsdp_model, auto_acc_model):
        input_data = [
            torch.ones([1, 16], dtype=torch.float32).to(device),
            torch.ones([1, 4], dtype=torch.float32).to(device),
        ]
        out = model(input_data)
        loss = loss_func(input_data, out)
        loss.backward()

    LARGE_FACTOR = 100
    for param in itertools.chain(fsdp_model.parameters(), auto_acc_model.parameters()):
        if param.grad is not None:  # gradients may be `None` for `use_orig_params=True`
            param.grad *= LARGE_FACTOR

    max_norm = 2.5
    norm_type = 2
    auto_acc_grad_norm = auto_acc_clip_grad_norm(auto_acc_model, max_norm=max_norm, norm_type=norm_type)
    if torch_version() >= (1, 12, 1) and torch_version() <= (1, 13, 1):
        from torch.testing._internal.common_fsdp import _collect_total_grad_norm_fsdp

        # in torch 1.x, fsdp_model.clip_grad_norm_ returns None.
        fsdp_total_norm = _collect_total_grad_norm_fsdp(fsdp_model, norm_type, rank)
    elif torch_version() >= (2, 0, 0):
        fsdp_total_norm = fsdp_model.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)
    assert torch.equal(auto_acc_grad_norm, fsdp_total_norm)

    dist.destroy_process_group()


def _fsdp_norm_utils(rank, world_size, free_port):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    local_model = ToyModel(use_custom_module=True)
    atorch_wrap_cls = {
        ToyCustomModule,
    }
    ref_local_model = deepcopy(local_model).to(device)
    fsdp_strategy = [
        ("parallel_mode", ([("data", torch.distributed.get_world_size())], None)),
        ("fsdp", {"atorch_wrap_cls": atorch_wrap_cls, "use_orig_params": True, "sync_module_states": True}),
    ]
    status, result, _ = auto_accelerate(
        model=local_model,
        load_strategy=fsdp_strategy,
        ignore_dryrun_on_load_strategy=True,
    )
    assert status is True
    auto_acc_model = result.model

    for model in (ref_local_model, auto_acc_model):
        batch_dim = 1 if model is auto_acc_model else world_size
        input_data = [
            torch.ones([batch_dim, 16], dtype=torch.float32).to(device),
            torch.ones([batch_dim, 4], dtype=torch.float32).to(device),
        ]
        out = model(input_data)
        loss = loss_func(input_data, out)
        loss.backward()

    LARGE_FACTOR = 100
    for param in itertools.chain(ref_local_model.parameters(), auto_acc_model.parameters()):
        if param.grad is not None:  # gradients may be `None` for `use_orig_params=True`
            param.grad *= LARGE_FACTOR

    norm_type = 2

    total_param_norm, total_grad_num_zeros = fsdp_param_norm_and_grad_num_zero(auto_acc_model, norm_type=norm_type)
    ref_total_param_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(param.detach(), norm_type, dtype=torch.float32)
                for param in ref_local_model.parameters()
                if param.numel() > 0
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )

    def _count_grad_zero(params):
        grad_num_zeros = 0
        for param in params:
            if param.grad is not None:
                grad = param.grad.detach()
                grad_num_zeros += grad.numel() - torch.count_nonzero(grad)
        return grad_num_zeros

    ref_total_grad_num_zeros = _count_grad_zero(ref_local_model.parameters())
    if torch.distributed.get_rank() == 0:
        assert (
            torch.isclose(total_param_norm, ref_total_param_norm)
            and total_grad_num_zeros.item() == ref_total_grad_num_zeros.item()
        )
    # print(f"total_param_norm: {total_param_norm}, total_grad_num_zeros: {total_grad_num_zeros}, "
    #       f"ref_total_param_norm: {ref_total_param_norm}, ref_total_grad_num_zeros: {ref_total_grad_num_zeros}")

    grad_norms = fsdp_grad_norm_per_param(auto_acc_model, 1, norm_type=norm_type)
    ref_grad_norms = {
        name: torch.linalg.vector_norm(param.grad.detach(), norm_type, dtype=torch.float32)
        if param.grad is not None
        else torch.tensor(0).to(torch.cuda.current_device())
        for name, param in ref_local_model.named_parameters()
    }
    if torch.distributed.get_rank() == 0:
        assert set(grad_norms.keys()) == set(ref_grad_norms.keys())
        for k in grad_norms:
            assert torch.isclose(grad_norms[k], ref_grad_norms[k])
    # print(f"grad_norms: {grad_norms}")
    # print(f"ref_grad_norms: {ref_grad_norms}")

    dist.destroy_process_group()


def is_tensor_parallel_parameter(param):
    return hasattr(param, "is_tensor_parallel") and param.is_tensor_parallel


def clip_grad_norm_for_ut(parameters, max_norm, norm_type=2, tp_group=None):
    """Adapted from deepspeed"""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # Take max across all GPUs.
        if tp_group is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.MAX, group=tp_group)
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0
        for p in parameters:
            if tp_group is not None:
                if (torch.distributed.get_rank(tp_group) == 0) or is_tensor_parallel_parameter(p):
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item() ** norm_type
            else:
                param_norm = p.grad.data.float().norm(norm_type)
                total_norm += param_norm.item() ** norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        if tp_group is not None:
            dist.all_reduce(total_norm_cuda, op=dist.ReduceOp.SUM, group=tp_group)
        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm


def _tp_strategy_clip_grad_norm(rank, world_size, free_port):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    init_distributed(backend="nccl")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(torch.device(f"cuda:{rank}"))
    local_model = ToyModel(use_custom_module=True)
    tp_strategy = [("parallel_mode", ([("tensor", torch.distributed.get_world_size())], None)), "tensor_parallel"]
    status, result, _ = auto_accelerate(
        model=local_model,
        load_strategy=tp_strategy,
        ignore_dryrun_on_load_strategy=True,
        sample_batch=[
            torch.ones([1, 16], dtype=torch.float32).to(device),
            torch.ones([1, 4], dtype=torch.float32).to(device),
        ],
        batch_size=1,
    )
    assert status is True
    auto_acc_model = result.model
    max_norm = 2.5
    norm_type = 2
    auto_acc_grad_norm = auto_acc_clip_grad_norm(auto_acc_model, max_norm=max_norm, norm_type=norm_type)
    tp_group = parallel_group("tensor")
    ds_result = clip_grad_norm_for_ut(
        auto_acc_model.parameters(), max_norm=max_norm, norm_type=norm_type, tp_group=tp_group
    )
    assert auto_acc_grad_norm.cpu().item() == ds_result
    reset_distributed()


class TestClipGradNorm(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "No gpu available for cuda tests",
    )
    def test_fsdp_strategy_clip_grad_norm(self):
        world_size = 2
        mp.spawn(
            _fsdp_strategy_clip_grad_norm,
            args=(world_size, find_free_port()),
            nprocs=world_size,
            join=True,
            daemon=False,
            start_method="spawn",
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "No gpu available for cuda tests",
    )
    def test_tp_strategy_clip_grad_norm(self):
        world_size = 2
        mp.spawn(
            _tp_strategy_clip_grad_norm,
            args=(world_size, find_free_port()),
            nprocs=world_size,
            join=True,
            daemon=False,
            start_method="spawn",
        )


class TestFSDPNormUtils(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2 or torch_version() < (2, 0, 0),  # type: ignore
        "No gpu available for cuda tests, torch 2.0 needed for use_orig_param",
    )
    def test_fsdp_norm_utils(self):
        world_size = 2
        mp.spawn(
            _fsdp_norm_utils,
            args=(world_size, find_free_port()),
            nprocs=world_size,
            join=True,
            daemon=False,
            start_method="spawn",
        )


if __name__ == "__main__":
    unittest.main()
