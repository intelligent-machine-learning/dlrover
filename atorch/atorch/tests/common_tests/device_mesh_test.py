import os
import unittest

import torch
import torch.multiprocessing as mp

import atorch
from atorch.common.util_func import find_free_port

# from atorch.distributed import distributed as _distributed_context
from atorch.distributed.distributed import (
    create_parallel_group,
    destroy_parallel_group,
    get_device_mesh,
    local_rank,
    parallel_group,
)
from atorch.utils.version import torch_version

skip = None
if torch_version() >= (2, 4, 0):  # type: ignore
    from torch.distributed.device_mesh import DeviceMesh

    skip = False
else:
    DeviceMesh = object
    skip = True


def run_device_mesh(rank, world_size, multi_group=False):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    world_size = torch.distributed.get_world_size()

    if multi_group:
        mode = ([("data", 2), ("pipe", 2)], None)
    else:
        mode = ([("data", torch.distributed.get_world_size())], None)
    create_parallel_group(mode, use_device_mesh=True)

    torch.distributed.barrier()
    tensor = torch.ones([1], dtype=torch.float32, device=torch.device(type="cuda", index=local_rank()))

    torch.distributed.all_reduce(tensor, group=parallel_group("data"))
    if multi_group:
        torch.distributed.all_reduce(tensor, group=parallel_group("pipe"))

    destroy_parallel_group()


def run_device_mesh_multi_mesh(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    world_size = torch.distributed.get_world_size()

    mode1 = ([("data", 4)], None)
    create_parallel_group(mode1, use_device_mesh=True)

    mode2 = ([("expert", 2), ("expert_fsdp", 2)], None)
    create_parallel_group(mode2, use_device_mesh=True)

    assert isinstance(get_device_mesh("data"), DeviceMesh)
    assert isinstance(get_device_mesh("expert"), DeviceMesh)
    assert isinstance(get_device_mesh("expert_fsdp"), DeviceMesh)
    assert get_device_mesh("expert") == get_device_mesh("expert_fsdp")


class DeviceMeshTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2 or skip, "Requires 2 gpus.")
    def test_device_mesh_one_group(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NPROC_PER_NODE"] = str(world_size)
        os.environ["NVTE_TORCH_COMPILE"] = str(0)
        mp.spawn(
            run_device_mesh,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(torch.cuda.device_count() < 4 or skip, "Requires 4 gpus.")
    def test_device_mesh_multi_group(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NPROC_PER_NODE"] = str(world_size)
        os.environ["NVTE_TORCH_COMPILE"] = str(0)
        mp.spawn(
            run_device_mesh,
            args=(world_size, True),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

    @unittest.skipIf(torch.cuda.device_count() < 4 or skip, "Requires 4 gpus.")
    def test_device_mesh_multi_mesh(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NPROC_PER_NODE"] = str(world_size)
        os.environ["NVTE_TORCH_COMPILE"] = str(0)
        mp.spawn(
            run_device_mesh_multi_mesh,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""
