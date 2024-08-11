import os
import unittest

import torch
import torch.multiprocessing as mp

import atorch
from atorch.common.util_func import find_free_port
from atorch.utils.import_util import is_torch_npu_available
from atorch.utils.version import torch_version

if is_torch_npu_available():
    from atorch import npu  # noqa


def run_all_to_all_single(rank, world_size):
    # This is required because these functions calls directly to the .dist and needs
    # the world to be initialized
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)

    device = "cuda"
    row = world_size * (rank + 1) * (world_size + 1) / 2
    x = torch.ones(int(row), 5, device=device) * (rank + 1)
    x.requires_grad = True
    y = torch.empty_like(x)
    split_sizes = [(i + 1) * (rank + 1) for i in range(world_size)]
    torch.distributed.all_to_all_single(y, x, output_split_sizes=split_sizes, input_split_sizes=split_sizes)
    expected = []
    for idx, tensor in enumerate(torch.split(x, split_sizes)):
        expected.append(torch.full_like(tensor, (idx + 1)))
    expected = torch.cat(expected)
    print("rank:", rank, "y:", y)
    print("rank:", rank, "expected:", expected)
    assert torch.allclose(y, expected)

    atorch.reset_distributed()


@unittest.skipIf(
    not is_torch_npu_available() or torch.cuda.device_count() < 2 or torch_version() < (2, 0, 0),  # type: ignore
    "Must have at least 2 GPUs for expert parallel test",
)
class TestAllToAllSingle(unittest.TestCase):
    def test_all_to_all_single(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_all_to_all_single,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""
