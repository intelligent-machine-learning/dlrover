import unittest

import torch
from torch.utils.data import Dataset

from atorch.auto.model_context import ModelContext
from atorch.auto.opt_lib.tensor_parallel_optimization import TensorParallelOptimization
from atorch.distributed.distributed import destroy_parallel_group


class MyModule(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.layer = torch.nn.Linear(in_features, out_features, bias=bias)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features, bias=bias) for _ in range(16)])

    def forward(self, input_):
        data = torch.nn.functional.softmax(self.layer(input_[0]), dim=-1)
        for op in self.layers:
            data = op(data)
            data = torch.nn.functional.softmax(data, dim=-1)
        return data


class ToyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.rand((7, 32)), torch.randn(16)


def prepare_input(data, device):
    return data[0].to(device), data[1].to(device)


def create_model_context(data_size=16, batch_size=3):
    model = MyModule(32, 16, True)
    dataset = ToyDataset(data_size)
    model_context = ModelContext(
        model=model,
        dataset=dataset,
        prepare_input=prepare_input,
        dataloader_args={"batch_size": batch_size, "drop_last": True},
    )
    return model_context


def _run_tensor_parallel_optimization(num_nodes=1, num_devices_per_node=2):
    # This is mock test for TensorParallelOptimization
    # Test to make sure TP can generate a replacement_map
    torch.manual_seed(42)

    model_context = create_model_context()

    parallel_optimization = TensorParallelOptimization(
        shard_planner="base",
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        tracer_backend="meta_fx",
        prop_mode="meta_tracer",
    )

    status, best_config, model_context = parallel_optimization.tune(model_context, {"tp_ranks": [0, 1]}, [])
    assert "replacement_map" in best_config and status


class TestTensorParallelOptimization(unittest.TestCase):
    def tearDown(self):
        destroy_parallel_group()
        return super().tearDown()

    def test_tensor_parallel_optimization(self):
        _run_tensor_parallel_optimization()


if __name__ == "__main__":
    unittest.main()
