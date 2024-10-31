import os
import unittest

import torch
import torch.multiprocessing as mp
from model_registry import MultiMLP, create_pipe_group

import atorch
from atorch.common.util_func import find_free_port
from atorch.pipeline_parallel.pipe_stage import PipeStage
from atorch.utils.version import torch_version

skip = None
if torch_version() >= (2, 4, 0):  # type: ignore
    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    skip = False
else:
    PipelineStage, ScheduleGPipe, = (  # type: ignore
        None,
        None,
    )
    skip = True


d_hid = 512
batch_size = 256
chunks = 4


def run_manual(rank, world_size, use_atorch=False):
    create_pipe_group(rank)

    device = torch.cuda.current_device()
    full_mod = MultiMLP(d_hid, n_layers=world_size)
    full_mod.to(device)
    stage_mod = full_mod.get_submodule(f"layers.{rank}")

    x = torch.randn(batch_size, d_hid, device=device)

    if use_atorch:
        activation_mapping = [("x", 0)]
        batch_mapping = [("x", 0)]
        io_mapping = (activation_mapping, batch_mapping)
        stage = PipeStage(stage_mod, rank, world_size, device, io_mapping)
    else:
        stage = PipelineStage(
            stage_mod,
            rank,
            world_size,
            device,
            input_args=x.chunk(chunks)[0],
        )

    # Attach to a schedule
    schedule = ScheduleGPipe(stage, chunks)

    # Run
    def _run_step(x):
        if rank == 0:
            return schedule.step(x)
        else:
            return schedule.step()

    out = _run_step(x)
    # Last rank checks result
    if rank == world_size - 1:
        ref_out = full_mod(x)
        torch.allclose(out, ref_out)

    atorch.reset_distributed()


@unittest.skipIf(torch.cuda.device_count() < 2 or skip, "Requires 2 gpus.")
class StageTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_torch_manual_pipe_stage(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            run_manual,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )

    def test_atorch_manual_pipe_stage(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            run_manual,
            args=(world_size, True),
            nprocs=world_size,
            join=True,
        )
