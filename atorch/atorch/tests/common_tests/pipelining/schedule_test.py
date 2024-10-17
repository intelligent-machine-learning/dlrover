import copy
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from model_registry import MultiMLP, create_pipe_group

import atorch
from atorch.common.util_func import find_free_port
from atorch.pipeline_parallel.pipe_stage import PipeStage
from atorch.utils.version import torch_version

skip = None
if torch_version() >= (2, 4, 0):  # type: ignore
    from torch.distributed.pipelining import PipelineStage, Schedule1F1B, ScheduleInterleaved1F1B

    skip = False
else:
    skip = True
    PipelineStage, Schedule1F1B, ScheduleInterleaved1F1B = None, None, None  # type: ignore


d_hid = 512
batch_size = 256


def run_grad_with_manual(rank, world_size, use_atorch=False):
    create_pipe_group(rank)

    device = torch.cuda.current_device()

    full_mod = MultiMLP(d_hid, n_layers=world_size)
    full_mod.to(device)

    ref_mod = copy.deepcopy(full_mod)
    x = torch.randn(batch_size, d_hid, device=device)
    with torch.no_grad():
        y = ref_mod(x)
        # Add a small perturbation
        target = y + torch.randn(batch_size, d_hid, device=device)

    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Run reference
    for _ in range(2):
        ref_mod.zero_grad()
        ref_out = ref_mod(x)
        ref_loss = loss_fn(ref_out, target)
        ref_loss.backward()

    # Get a submodule, e.g. `layers.0` or `layers.1`
    submod_name = f"layers.{rank}"
    stage_module = full_mod.get_submodule(submod_name)
    chunks = 8
    # Create a pipeline stage to wrap that submodule

    if use_atorch:
        activation_mapping = [("x", 0)]
        batch_mapping = [("x", 0)] if rank == 0 else None
        io_mapping = (activation_mapping, batch_mapping)
        stage = PipeStage(stage_module, rank, world_size, device, io_mapping)
    else:
        stage = PipelineStage(
            stage_module,
            rank,
            world_size,
            device,
            input_args=x.chunk(chunks)[0],
        )

    # Attach to a schedule
    schedule = Schedule1F1B(stage, chunks, loss_fn=loss_fn)

    # Run
    for _ in range(2):
        # Zero gradients
        stage_module.zero_grad()
        if rank == 0:
            schedule.step(x)
        elif rank == world_size - 1:
            losses = []
            out = schedule.step(target=target, losses=losses)
        else:
            schedule.step()

    dist.barrier()

    # Last rank checks result
    if rank == world_size - 1:
        # Check output
        torch.allclose(out, ref_out)
        # Check loss
        # Since the reduction used in the loss function above is "sum", we use
        # "sum" here to reduce microbatch losses into a single value too.
        pipe_loss = sum(losses)
        torch.allclose(pipe_loss, ref_loss)

    # Every rank checks gradients
    ref_submod = ref_mod.get_submodule(submod_name)
    for name, p in stage_module.named_parameters():
        ref_p = ref_submod.get_parameter(name)
        try:
            torch.allclose(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
        except AssertionError:
            print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
            raise

    atorch.reset_distributed()


def run_grad_with_manual_interleaved(rank, world_size, use_atorch=False):
    create_pipe_group(rank)

    device = torch.cuda.current_device()

    stages_per_rank = 2
    n_stages = stages_per_rank * world_size
    full_mod = MultiMLP(d_hid, n_layers=n_stages)
    full_mod.to(device)

    ref_mod = copy.deepcopy(full_mod)
    x = torch.randn(batch_size, d_hid, device=device)
    with torch.no_grad():
        y = ref_mod(x)
        # Add a small perturbation
        target = y + torch.randn(batch_size, d_hid, device=device)

    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Run reference
    for _ in range(2):
        ref_mod.zero_grad()
        ref_out = ref_mod(x)
        ref_loss = loss_fn(ref_out, target)
        ref_loss.backward()

    # Get a submodule, e.g. `layers.0` or `layers.1`
    stage_indices = [rank + i * world_size for i in range(stages_per_rank)]
    print(f"Rank {rank} stages: {stage_indices}")
    submod_names = [f"layers.{i}" for i in stage_indices]
    stage_modules = [full_mod.get_submodule(submod_name) for submod_name in submod_names]
    # Create a pipeline stage to wrap that submodule
    chunks = 8
    input_args = x.chunk(chunks)[0]

    if use_atorch:
        activation_mapping = [("x", 0)]
        batch_mapping = [("x", 0)]
        stages = []
        for stage_module, stage_idx in zip(stage_modules, stage_indices):
            batch_mapping = batch_mapping if stage_idx == 0 else None
            io_mapping = (activation_mapping, batch_mapping)
            stage = PipeStage(stage_module, stage_idx, n_stages, device, io_mapping)
            stages.append(stage)
    else:
        stages = [
            PipelineStage(
                stage_module,
                stage_idx,
                n_stages,
                device,
                input_args=input_args,
            )
            for stage_module, stage_idx in zip(stage_modules, stage_indices)
        ]

    # Attach to a schedule
    schedule = ScheduleInterleaved1F1B(stages, chunks, loss_fn=loss_fn)

    # Run
    for _ in range(2):
        # Zero gradients
        for stage_module in stage_modules:
            stage_module.zero_grad()
        if rank == 0:
            schedule.step(x)
        elif rank == world_size - 1:
            losses = []
            out = schedule.step(target=target, losses=losses)
        else:
            schedule.step()

    dist.barrier()

    # Last rank checks result
    if rank == world_size - 1:
        # Check output
        torch.allclose(out, ref_out)
        # Check loss
        # Since the reduction used in the loss function above is "sum", we use
        # "sum" here to reduce microbatch losses into a single value too.
        pipe_loss = sum(losses)
        torch.allclose(pipe_loss, ref_loss)

    # Every rank checks gradients
    for stage_module, submod_name in zip(stage_modules, submod_names):
        # Get corresponding submodule from reference model
        ref_submod = ref_mod.get_submodule(submod_name)
        # Check gradients per parameter
        for name, p in stage_module.named_parameters():
            ref_p = ref_submod.get_parameter(name)
            try:
                torch.allclose(p.grad, ref_p.grad, rtol=1e-5, atol=4e-5)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

    atorch.reset_distributed()


@unittest.skipIf(torch.cuda.device_count() < 2 or skip, "Requires 2 gpus.")
class ScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_grad_with_manual(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            run_grad_with_manual,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    def test_grad_with_manual_atorch(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            run_grad_with_manual,
            args=(world_size, True),
            nprocs=world_size,
            join=True,
        )

    def test_grad_with_manual_interleaved(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            run_grad_with_manual_interleaved,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    def test_grad_with_manual_interleaved_atorch(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            run_grad_with_manual_interleaved,
            args=(world_size, True),
            nprocs=world_size,
            join=True,
        )
