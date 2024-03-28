import os
import unittest

import torch
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _get_default_group

import atorch
from atorch.common.util_func import set_sync_bn_pg
from atorch.distributed.distributed import create_parallel_group, parallel_group


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    if torch.cuda.is_available():
        atorch.init_distributed("nccl")
    else:
        atorch.init_distributed("gloo")


def _test_set_sync_batch_norm_process_group(rank, world_size):
    init_dist(rank, world_size)
    model_parallel_size = 2
    data_parallel_size = 1
    parallel_config = ([("model", model_parallel_size), ("data", data_parallel_size)], None)
    create_parallel_group(parallel_config)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{atorch.local_rank()}")
    else:
        device = torch.device("cpu")

    # Network with nn.BatchNorm layer
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 4),
        torch.nn.BatchNorm1d(4),
    ).to(device)
    sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, _get_default_group())

    for _, child in sync_bn_module.named_modules():
        if isinstance(child, torch.nn.SyncBatchNorm):
            assert child.process_group.size() == world_size

    set_sync_bn_pg(sync_bn_module, parallel_group("data"))

    for _, child in sync_bn_module.named_modules():
        if isinstance(child, torch.nn.SyncBatchNorm):
            assert child.process_group.size() == data_parallel_size

    atorch.reset_distributed()


class TestReshardingOperator(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "No gpu available for cuda tests",
    )
    def test_syncbatchnorm_process_group(self):
        world_size = 2
        mp.spawn(
            _test_set_sync_batch_norm_process_group,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
