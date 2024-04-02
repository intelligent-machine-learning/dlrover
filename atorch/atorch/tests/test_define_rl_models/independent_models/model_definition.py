import torch
import torch.nn as nn


class FakeActor(nn.Module):
    def __init__(self, features_in=10, features_out=10):
        super().__init__()
        self.linear = torch.nn.Linear(features_in, features_out)

    def forward(self, x):
        return self.linear(x)


class FakeCritic(nn.Module):
    def __init__(self, dims_in=10, dims_out=10):
        super().__init__()
        self.linear = torch.nn.Linear(dims_in, dims_out)

    def forward(self, x):
        return self.linear(x)


class FakeRewardModel(nn.Module):
    def __init__(self, dims_in=10, dims_out=10):
        super().__init__()
        self.linear = torch.nn.Linear(dims_in, dims_out)

    def forward(self, x):
        return self.linear(x)


class FakeRefModel(nn.Module):
    def __init__(self, dims_in=10, dims_out=10):
        super().__init__()
        self.linear = torch.nn.Linear(dims_in, dims_out)

    def forward(self, x):
        return self.linear(x)
