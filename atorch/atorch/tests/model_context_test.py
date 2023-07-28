import os
import unittest

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import atorch
from atorch.auto.model_context import ModelContext
from atorch.data import ShmDataloader
from atorch.distributed.distributed import _DistributedContext, create_parallel_group
from atorch.tests.test_utils import run_multi_process_init_distributed
from atorch.tests.toy_module import ToyDataset, create_model_context, run_train


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

    @unittest.skipIf(torch.cuda.is_available(), "test only on cpu")
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


class ModelContextWrapperTest(unittest.TestCase):
    def setUp(self):
        self.context = create_model_context(data_size=2, batch_size=1)

    def test_adjust_ddp_and_zero2_wrapper(self):
        self.context.add_wrapper("zero2", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("ddp", None, None, is_pre_wrapper=False)
        wrappers = self.context.post_wrappers
        self.assertIn("zero2", wrappers)
        self.assertIn("ddp", wrappers)
        self.context.adjust_wrappers()
        wrappers = self.context.post_wrappers
        self.assertIn("zero2", wrappers)
        self.assertNotIn("ddp", wrappers)

    def test_adjust_amp_apex_and_ddp_wrapper(self):
        self.context.add_wrapper("ddp", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("amp_apex_o2", None, None, is_pre_wrapper=False)
        self.context.adjust_wrappers()
        wrapper_names = [name for name, _ in self.context.post_wrappers.items()]
        ddp_index = wrapper_names.index("ddp")
        amp_apex_o2_index = wrapper_names.index("amp_apex_o2")
        self.assertLess(amp_apex_o2_index, ddp_index)

    def test_adjust_amp_apex_and_zero2_wrapper(self):
        self.context.add_wrapper("zero2", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("amp_apex_o2", None, None, is_pre_wrapper=False)
        self.context.adjust_wrappers()
        wrapper_names = [name for name, _ in self.context.post_wrappers.items()]
        zero2_index = wrapper_names.index("zero2")
        amp_apex_o2_index = wrapper_names.index("amp_apex_o2")
        self.assertLess(amp_apex_o2_index, zero2_index)

    def test_adjust_amp_apex_zero2_ddp_wrapper(self):
        self.context.add_wrapper("zero2", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("ddp", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("opt0", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("opt1", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("opt2", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("amp_apex_o1", None, None, is_pre_wrapper=False)
        self.context.add_wrapper("opt3", None, None, is_pre_wrapper=False)
        self.context.adjust_wrappers()
        wrapper_names = [name for name, _ in self.context.post_wrappers.items()]
        self.assertNotIn("ddp", wrapper_names)
        zero_index = wrapper_names.index("zero2")
        amp_apex_o2_index = wrapper_names.index("amp_apex_o1")
        wrappers_order = [
            wrapper_names.index("opt0"),
            wrapper_names.index("opt1"),
            wrapper_names.index("opt2"),
            amp_apex_o2_index,
            zero_index,
            wrapper_names.index("opt3"),
        ]
        self.assertLess(amp_apex_o2_index, zero_index)
        self.assertEqual(wrappers_order, [0, 1, 2, 3, 4, 5])


def use_shm_dataloader_func():
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
    dataloader_args = {"batch_size": batch_size, "drop_last": True}
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


class ModelContextShmDataloaderTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.is_available(), "Skip on gpu as cpu test covers it.")
    def test_use_shm_dataloader(self):
        code_path = os.path.abspath(__file__)
        run_multi_process_init_distributed(nproc=2, training_script=code_path)


if __name__ == "__main__":
    use_shm_dataloader_func()
