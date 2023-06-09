import unittest

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import atorch
from atorch.distributed.distributed import _DistributedContext
from atorch.tests.test_utils import run_multi_process_init_distributed
from atorch.tests.toy_module import create_model_context, run_train


class ModelContextTest(unittest.TestCase):
    def run_test_with_device(self, device, data_size=16, batch_size=2):
        context = create_model_context(data_size=data_size, batch_size=batch_size)
        context.update_dataloader()
        context.update_optim()
        context.model.to(device)
        num = run_train(
            context.model, context.dataloader, context.optim, context.prepare_input, context.loss_func, device
        )
        self.assertEqual(num, data_size / batch_size)
        context.update_dataloader(extra_args={"batch_size": 4})

        def multi_output_loss_func(input, output):
            loss = context.loss_func(input, output)
            return loss, True

        num = run_train(
            context.model, context.dataloader, context.optim, context.prepare_input, multi_output_loss_func, device
        )
        self.assertEqual(num, data_size / 4)

    @unittest.skipIf(not torch.cuda.is_available(), "No gpu available for cuda tests")
    def test_gpu(self):
        device = "cuda"
        self.run_test_with_device(device)

    def test_cpu(self):
        device = "cpu"
        self.run_test_with_device(device)

    def test_optim_param_func(self):
        context = create_model_context(data_size=16, batch_size=2, use_optim_param_func=True)
        context.update_dataloader()
        optimizer = context.create_optim()
        weight_decay_set = set()
        for group in optimizer.param_groups:
            weight_decay_set.add(group["weight_decay"])
        self.assertEqual(weight_decay_set, {0.0, 0.01})

    def test_find_unused_parameters(self):
        context = create_model_context(data_size=16, batch_size=2)
        self.assertFalse(context.find_unused_parameters)
        context = create_model_context(data_size=16, batch_size=2, extra_args={"find_unused_parameters": True})
        self.assertTrue(context.find_unused_parameters)

    def test_dataloader_shuffle(self):
        class TestDataset(Dataset):
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return torch.Tensor([idx]), torch.Tensor([idx])

        dataset = TestDataset(16)
        atorch.init_distributed("gloo")
        context = create_model_context(
            data_size=16, batch_size=2, dataset=dataset, distributed_sampler_cls=DistributedSampler
        )
        # hack to pretend that data parallel size is 2, rank = 1
        _DistributedContext.PARALLEL_GROUP_SIZE = {}
        _DistributedContext.PARALLEL_GROUP_SIZE["data"] = 2
        _DistributedContext.RANK = 1
        dataloader = context.create_dataloader()
        data_e0 = [data for data in dataloader]
        data_e0b = [data for data in dataloader]
        self.assertEqual(data_e0, data_e0b)
        dataloader.set_epoch(1)
        data_e1 = [data for data in dataloader]
        self.assertNotEqual(data_e0, data_e1)
        atorch.reset_distributed()


run_use_shm_dataloader_code = """
import os
import torch
import atorch
import numpy as np
from atorch.distributed.distributed import create_parallel_group
from atorch.data import ShmDataloader
from atorch.auto.model_context import ModelContext
from atorch.tests.toy_module import ToyDataset

if __name__ == "__main__":
    if torch.cuda.is_available():
        atorch.init_distributed("nccl")
    else:
        atorch.init_distributed("gloo")

    parallel_config = ([("model", 2), ("data", 1)], None)
    create_parallel_group(parallel_config)

    os.environ["ENABLE_SHM_DATALOADER"] = "True"
    d_size = 4
    batch_size = 2
    dataset = ToyDataset(size=d_size, data_size=2, output_size=2)
    dataloader_args={"batch_size": batch_size, "drop_last": True}
    context = ModelContext(
        dataset=dataset,
        dataloader_args=dataloader_args,
    )

    dataloader = context.create_dataloader()
    assert isinstance(dataloader, ShmDataloader)
    assert dataloader.shm_context.rank == atorch.distributed.rank()
    for _ in range(2):
        count = 0
        total_0 = total_1 = 0
        for batch in dataloader:
            count += 1
            for i in range(batch_size):
                total_0 += batch[0][i][0].item()
                total_1 += batch[1][i][0].item()
            torch.distributed.barrier()
        assert count == d_size // batch_size
        assert total_0 == d_size * (d_size - 1) // 2
        assert total_1 == d_size
        assert count == d_size / batch_size


    atorch.reset_distributed()
"""


class ModelContextShmDataloaderTest(unittest.TestCase):
    def test_use_shm_dataloader(self):
        codes = run_use_shm_dataloader_code
        run_multi_process_init_distributed(codes, nproc=2)


if __name__ == "__main__":
    unittest.main()
