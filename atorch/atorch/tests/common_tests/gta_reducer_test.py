import os
import random
import unittest
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing_extensions import Literal

from atorch.local_sgd.reduce_methods import GTAReducer, LinearReducer
from atorch.local_sgd.reduce_methods.sparsify import SparsificationMethod
from atorch.tests.utils.test_utils import find_free_port


def seed_everything(seed=42):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # Numpy
    torch.manual_seed(seed)  # PyTorch

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_gta_correctness(
    rank,
    world_size,
    answers=None,
    weighted=False,
    method: Literal["sum", "count"] = "sum",
    normalize=True,
    sparsification_method: Optional[SparsificationMethod] = None,
):
    seed_everything(42)

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    dp_group = dist.new_group(list(range(world_size)), backend="nccl")
    reducer = GTAReducer(
        process_group=dp_group,
        consensus_method=method,
        sparsification_method=sparsification_method,
        normalize=normalize,
    )
    local_tensor = torch.tensor([(rank + 1) * (-1) ** (rank)] * world_size, device=f"cuda:{rank}", dtype=torch.float32)
    answer = answers[(weighted, method, normalize)].to(local_tensor.device) if answers is not None else None
    kwargs = {}
    if weighted:
        kwargs["weight"] = rank + 1
    reducer.reduce_tensor(tensors=local_tensor, **kwargs)
    if answer is not None:
        assert torch.allclose(local_tensor, answer, atol=1e-5)


def check_linear_correctness(rank, world_size, answer):
    seed_everything(42)

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

    dp_group = dist.new_group(list(range(world_size)), backend="nccl")
    reducer = LinearReducer(process_group=dp_group, normalize=True)
    local_tensor = torch.tensor([(rank + 1) * (-1) ** (rank)] * world_size, device=f"cuda:{rank}", dtype=torch.float32)
    answer = answer.to(local_tensor.device)
    reducer.reduce_tensor(tensors=local_tensor)
    assert torch.allclose(local_tensor, answer, atol=1e-5)


class TestReducer(unittest.TestCase):
    def tearDown(self):
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""

        return super().tearDown()

    def setUp(self):
        # set up distributed env first
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        self.world_size = 4
        self.answers = {
            (False, "sum", True): torch.tensor([-3, -3, -3, -3], dtype=torch.float32),
            (False, "count", True): torch.tensor([2, 2, 2, 2], dtype=torch.float32),
            (True, "sum", True): torch.tensor([-10 / 3, -10 / 3, -10 / 3, -10 / 3], dtype=torch.float32),
            (True, "count", True): torch.tensor([5 / 2, 5 / 2, 5 / 2, 5 / 2], dtype=torch.float32),
        }
        self.linear_answer = torch.tensor([-0.5, -0.5, -0.5, -0.5])

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Needs at least 4 gpu to run the reducer test",
    )
    def test_gta_sum_noramlize(self):
        mp.spawn(
            check_gta_correctness,
            args=(self.world_size, self.answers, False, "sum", True),
            nprocs=self.world_size,
            join=True,
        )

    # Don't check sparsification method's numerical correctness due to randomness
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Needs at least 4 gpu to run the reducer test",
    )
    def test_gta_sum_normalize_sparsify_magnitude(self):
        mp.spawn(
            check_gta_correctness,
            args=(self.world_size, None, False, "sum", True, "magnitude"),
            nprocs=self.world_size,
            join=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Needs at least 4 gpu to run the reducer test",
    )
    def test_gta_sum_normalize_sparsify_random(self):
        mp.spawn(
            check_gta_correctness,
            args=(self.world_size, None, False, "sum", True, "random"),
            nprocs=self.world_size,
            join=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Needs at least 4 gpu to run the reducer test",
    )
    def test_gta_sum_normalize_sparsify_rescaled_random(self):
        mp.spawn(
            check_gta_correctness,
            args=(self.world_size, None, False, "sum", True, "rescaled_random"),
            nprocs=self.world_size,
            join=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Needs at least 4 gpu to run the reducer test",
    )
    def test_gta_count_noramlize(self):
        mp.spawn(
            check_gta_correctness,
            args=(self.world_size, self.answers, False, "count", True),
            nprocs=self.world_size,
            join=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Needs at least 4 gpu to run the reducer test",
    )
    def test_gta_weighted_sum_noramlize(self):
        mp.spawn(
            check_gta_correctness,
            args=(self.world_size, self.answers, True, "sum", True),
            nprocs=self.world_size,
            join=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Needs at least 4 gpu to run the reducer test",
    )
    def test_gta_weighted_count_noramlize(self):
        mp.spawn(
            check_gta_correctness,
            args=(self.world_size, self.answers, True, "count", True),
            nprocs=self.world_size,
            join=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Needs at least 4 gpu to run the reducer test",
    )
    def test_linear(self):
        mp.spawn(
            check_linear_correctness, args=(self.world_size, self.linear_answer), nprocs=self.world_size, join=True
        )


if __name__ == "__main__":
    unittest.main()
