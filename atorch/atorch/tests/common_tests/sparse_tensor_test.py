import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from atorch.common.util_func import find_free_port
from atorch.utils.sparse import all_reduce_sparse


def run_all_reduce_sparse(rank, size, backend="gloo"):
    """Distributed function to be implemented later."""
    dist.init_process_group(backend, rank=rank, world_size=size)
    if rank == 0:
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.1, 1.2]]).to_sparse(2)
        result = all_reduce_sparse(data)
    else:
        data = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]).to_sparse(2)
        result = all_reduce_sparse(data)

    indices = torch.tensor([[1, 1], [1, 2]])
    values = torch.tensor([1.1000, 3.2000])
    tensor_size = torch.tensor([2, 3])
    assert torch.equal(result.indices(), indices)
    assert torch.equal(result.values(), values)
    assert torch.equal(torch.tensor(result.size()), tensor_size)


class SparseTensorCommunicationTest(unittest.TestCase):
    @unittest.skipIf(True, "Failed on gpu")
    def test_all_reduce_sparse(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        world_size = 2
        mp.spawn(run_all_reduce_sparse, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    unittest.main()
