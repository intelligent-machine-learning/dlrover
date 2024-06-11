import unittest

import torch
from torch.nn import MSELoss
from torch.utils.data import Dataset

from atorch.auto.model_context import ModelContext
from atorch.auto.opt_lib.shard_planners.dim_planner import DimPlanner
from atorch.utils.version import torch_version

skip = False
try:
    from pippy.IR import LossWrapper  # noqa # type: ignore
except ImportError:
    skip = True


class FakeDeviceContext(object):
    """
    Device context contains compute resources information below.
        number of nodes
        number of logical cpu cores per node
        gpu model
        gpu memory(B)
        number of gpus per node
        total gpus of the training job
    """

    def __init__(self):
        self.intra_node_bandwidth = 2**4
        self.inter_node_bandwidth = 2**3
        self.fp32_flops = 2**4
        self.gpu_memory = 2**13


class MyModule(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.layer = torch.nn.Linear(in_features, out_features, bias=bias)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(out_features, out_features, bias=bias) for _ in range(3)])

    def forward(self, input_):
        data = torch.nn.functional.gelu(self.layer(input_[0]))
        for op in self.layers:
            data = op(data)
        return data


class ToyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.ones((4, 8)), torch.zeros((4, 8))


def prepare_input(data, device):
    return data[0].to(device), data[1].to(device)


def my_loss_func(data, outputs):
    loss_fct = MSELoss()
    loss = loss_fct(outputs.view(-1), data[-1].view(-1))
    return loss


def create_model_context(data_size=512, batch_size=16, loss_func=None):
    model = MyModule(8, 8, True)
    dataset = ToyDataset(data_size)
    model_context = ModelContext(
        model=model,
        optim_func=torch.optim.SGD,
        dataset=dataset,
        prepare_input=prepare_input,
        dataloader_args={"batch_size": batch_size, "drop_last": True},
        optim_args={"lr": 0.001},
        loss_func=loss_func,
    )
    return model_context


def run_dim_planner(num_nodes, num_devices_per_node, loss_func):
    model_context = create_model_context(loss_func=loss_func)
    device_context = FakeDeviceContext()
    dim_planner = DimPlanner(
        num_nodes=num_nodes,
        num_devices_per_node=num_devices_per_node,
        use_fake_mode=False,
        device_context=device_context,
    )
    (
        optimal_tensor_size,
        optimal_pipe_size,
        optimal_data_size,
        insert_before_nodes,
    ) = dim_planner.generate_sharding_plan(model_context)
    assert len(insert_before_nodes) == optimal_pipe_size - 1
    assert optimal_tensor_size * optimal_pipe_size * optimal_data_size == num_nodes * num_devices_per_node


class TestDimPlanner(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch_version() < (2, 0, 0) or skip, "Test on GPU image"  # type: ignore
    )
    def test_dim_planner(self):
        run_dim_planner(2, 4, my_loss_func)


if __name__ == "__main__":
    unittest.main()
