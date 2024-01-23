import unittest

import torch

from atorch.utils.tracer import MetaTracer


# create a nested module tree
# to make sure control flow variables
# can be correctly propogated
# also test if leaf modules can be correctly registered
class Level1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.layer0(x)


class Level2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1a = Level1()
        self.layer1b = torch.nn.Linear(4, 4)

    def forward(self, x, y=None):
        if x is not None and y is not None:
            raise ValueError("x and y set simultaneously")
        if x is not None:
            return self.layer1a(x)
        else:
            return self.layer1b(y)


class Level3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer2 = Level2()

    def forward(self, x, y=None):
        return self.layer2(x, y)


class Level4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer3 = Level3()

    def forward(self, x, y=None):
        return torch.nn.functional.relu(self.layer3(x, y))


class TestTracer(unittest.TestCase):
    def test_tracer(self):
        x = torch.rand((3, 4))
        model = Level4()
        tracer = MetaTracer()
        tracer.register_leaf_modules([Level1, Level3])
        traced_graph = tracer.trace(model, {"x": x})
        traced = torch.fx.GraphModule(model, traced_graph)
        # traced.device = model.device
        normal_output = model(x)
        test_output = traced(x)
        self.assertEqual(normal_output.shape, test_output.shape)
        for i in range(normal_output.shape[0]):
            for j in range(normal_output.shape[1]):
                self.assertEqual(float(normal_output[i][j]), float(test_output[i][j]))
