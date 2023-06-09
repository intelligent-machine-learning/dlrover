import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from atorch.auto.model_context import ModelContext


class ToyCustomModule(nn.Module):
    def __init__(self, in_features=16, out_features=4):
        super().__init__()
        self.linears = torch.nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(8)])

    def forward(self, inputs, test_kwargs=True):
        for op in self.linears:
            inputs = op(inputs)
        return inputs


class ToyModel(nn.Module):
    def __init__(self, in_features=16, out_features=4, use_custom_module=False):
        """
        Args:
            in_feature (int): size of input feature.
            out_feature (int): size of output feature.
        """
        super(ToyModel, self).__init__()
        self.use_custom_module = use_custom_module
        self.linear = torch.nn.Linear(in_features, out_features)
        if use_custom_module:
            self.linears = ToyCustomModule(in_features, out_features)
        else:
            self.linears = torch.nn.ModuleList([nn.Linear(out_features, out_features) for _ in range(8)])

    def forward(self, inputs):
        data = self.linear(inputs[0])
        if self.use_custom_module:
            data = self.linears(data, test_kwargs=True)
        else:
            for op in self.linears:
                data = op(data)
        return data


def optim_func(model_parameters, **kwargs):
    return optim.Adam(model_parameters, **kwargs)


def optim_param_func(model):
    no_decay = "bias"
    parameters = [
        {
            "params": [p for n, p in model.named_parameters() if no_decay not in n],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if no_decay in n],
            "weight_decay": 0.0,
        },
    ]
    return parameters


def loss_func(inputs, output):
    loss = nn.MSELoss()
    return loss(inputs[1], output)


def prepare_input(data, device):
    return data[0].to(device), data[1].to(device)


class ToyDataset(Dataset):
    def __init__(self, size, data_size=(16,), output_size=(4,)):
        """
        Args:
            size (int): the of samples.
            data_size (tuple): the shape of one input, data_size[-1] must match the in_features
                in ToyModule
            output_size (tuple): the shape of output, output_size[-1] must match the out_feautes
                in ToyModule
        """
        self.size = size
        self.data_size = data_size
        self.output_size = output_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return np.ones(self.data_size, dtype=np.float32) * idx, np.ones(self.output_size, dtype=np.float32)


def create_model_context(
    data_size=16,
    batch_size=2,
    use_optim_param_func=False,
    dataset=None,
    distributed_sampler_cls=None,
    use_custom_module=False,
    extra_args=None,
):
    user_defined_optim_param_func = optim_param_func if use_optim_param_func else None
    model = ToyModel(use_custom_module=use_custom_module)
    dataset = ToyDataset(data_size) if dataset is None else dataset
    model_context = ModelContext(
        model=model,
        optim_func=optim_func,
        dataset=dataset,
        loss_func=loss_func,
        prepare_input=prepare_input,
        optim_args={"lr": 0.001},
        optim_param_func=user_defined_optim_param_func,
        dataloader_args={"batch_size": batch_size, "drop_last": True, "shuffle": True},
        distributed_sampler_cls=distributed_sampler_cls,
        extra_args=extra_args,
    )
    return model_context


def run_train(model, dataloader, optim, prepare_input, loss_func, device="cpu", input_dtype=torch.float32):
    for idx, data in enumerate(dataloader):
        pdata = prepare_input(data, device)
        if input_dtype != torch.float32:
            pdata = pdata[0].to(input_dtype), pdata[1].to(input_dtype)
        optim.zero_grad()
        output = model(pdata)
        loss = ModelContext.get_loss_from_loss_func_output(loss_func(pdata, output))
        loss.backward()
        optim.step()
    return idx + 1
