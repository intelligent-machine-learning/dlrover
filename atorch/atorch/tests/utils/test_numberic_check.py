# coding=utf-8
import os
import shutil
import tempfile
import unittest

import torch

from atorch.normalization import LayerNorm
from atorch.utils.numberic_checker import module_numberic_checker, move_to_device


class LeNet5(torch.nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            torch.nn.Tanh(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=120, out_features=84),
            LayerNorm(84),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return logits, probs


class NumbericCheckTestcast(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        return super().setUp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        return super().tearDown()

    def __test_on_device(self, device):
        layer = LeNet5(5).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)
        saved_file = os.path.join(self.temp_dir, "test_inputs.pth")
        torch.save([x, layer.state_dict()], saved_file)

        with module_numberic_checker(layer, mode="save"):
            out = layer(x)

        # test load
        layer2 = LeNet5(5).to(device)
        loaded_x, layer_state_dict = torch.load(saved_file)
        layer_state_dict = move_to_device(layer_state_dict, device)
        loaded_x = move_to_device(loaded_x, device)
        with module_numberic_checker(layer2, mode="load") as no_allclose:
            self.assertEqual(len(no_allclose), 0)
            layer2(loaded_x)
        self.assertNotEqual(len(no_allclose), 0)
        for func_name, err_msg in no_allclose.items():
            print(func_name, err_msg)
        layer2.load_state_dict(layer_state_dict)

        with module_numberic_checker(layer2, mode="load") as no_allclose:
            self.assertEqual(len(no_allclose), 0)
            out2 = layer2(loaded_x)
        self.assertEqual(len(no_allclose), 0)
        torch.testing.assert_close(out2, out)

    def test_cpu(self):
        device = "cpu"
        self.__test_on_device(device)

    @unittest.skipIf(not torch.cuda.is_available(), "no cuda device")
    def test_gpu(self):
        device = "cuda"
        self.__test_on_device(device)
