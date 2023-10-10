import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import create_parallel_group, parallel_group, parallel_group_and_ranks
from atorch.modules.distributed_modules.mappings_registry import get_communicator
from atorch.utils.sharding_spec import MeshShardingSpec

logger.setLevel("INFO")
os.environ["NCCL_DEBUG"] = "ERROR"


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = rank
    os.environ["RANK"] = rank
    os.environ["WORLD_SIZE"] = world_size
    os.environ["NPROC_PER_NODE"] = world_size
    atorch.init_distributed("nccl")
    torch.cuda.device(atorch.local_rank())
    parallel_config = ([("model", world_size)], None)
    create_parallel_group(parallel_config)


def _run_sequence_partition_to_feature_partition(rank, world_size):
    init_dist(rank, world_size)

    device = torch.device("cuda:{}".format(atorch.local_rank()))
    torch.cuda.set_device(device)

    pg = parallel_group("model")
    _, ranks = parallel_group_and_ranks("model")

    prev_shard = MeshShardingSpec(
        dims=(1,),
        group=pg,
        ranks=ranks,
    )
    cur_shard = MeshShardingSpec(
        dims=(-1,),
        group=pg,
        ranks=ranks,
    )
    my_rank = dist.get_rank(pg)
    local_tensor = [list(range(my_rank, my_rank + world_size**2, world_size))]
    local_tensor = [local_tensor]
    local_tensor = torch.Tensor(local_tensor).contiguous()
    local_tensor.to(device)
    resharding_operator = get_communicator(prev_shard, cur_shard)
    resharded_tensor = resharding_operator(local_tensor)
    target_resharded_tensor = [[[my_rank + i] for i in range(world_size)]]
    target_resharded_tensor = torch.Tensor(target_resharded_tensor)
    assert torch.all(torch.isclose(resharded_tensor, target_resharded_tensor))

    atorch.reset_distributed()


def _run_data_partition_to_feature_partition(rank, world_size):
    init_dist(rank, world_size)

    device = torch.device("cuda:{}".format(atorch.local_rank()))
    torch.cuda.set_device(device)

    pg = parallel_group("model")
    _, ranks = parallel_group_and_ranks("model")

    prev_shard = MeshShardingSpec(
        dims=(0,),
        group=pg,
        ranks=ranks,
    )
    cur_shard = MeshShardingSpec(
        dims=(-1,),
        group=pg,
        ranks=ranks,
    )
    my_rank = dist.get_rank(pg)
    local_tensor = [list(range(my_rank, my_rank + world_size**2, world_size))]
    local_tensor = [local_tensor]
    local_tensor = torch.Tensor(local_tensor).contiguous()
    local_tensor.to(device)
    resharding_operator = get_communicator(prev_shard, cur_shard)
    resharded_tensor = resharding_operator(local_tensor)
    target_resharded_tensor = [[[my_rank + i]] for i in range(world_size)]
    target_resharded_tensor = torch.Tensor(target_resharded_tensor)
    assert torch.all(torch.isclose(resharded_tensor, target_resharded_tensor))

    atorch.reset_distributed()


class TestReshardingOperator(unittest.TestCase):
    @unittest.skipIf(True, "Failed on gpu")
    def test_sequence_partition_to_feature_partition(self):
        world_size = 2
        mp.spawn(
            _run_sequence_partition_to_feature_partition,
            args=(world_size),
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(True, "Failed on gpu")
    def test_data_partition_to_feature_partition(self):
        world_size = 2
        mp.spawn(
            _run_data_partition_to_feature_partition,
            args=(world_size),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
