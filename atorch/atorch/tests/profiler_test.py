import unittest

import torch

from atorch.normalization import LayerNorm
from atorch.utils import AProfiler


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


def within_range(val, target, tolerance):
    return abs(val - target) / target < tolerance


TOLERANCE = 0.05

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class TestProfiler(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_conv_profiler(self):
        model = LeNet5(10)
        model.to(device)
        batch_size = 1024

        total_steps = 20
        profile_step = 5

        input = torch.randn(batch_size, 1, 32, 32, device=device)

        prof = AProfiler(model)

        for step in range(total_steps):

            if step == profile_step:
                prof.start_profile()

            model(input)

            if step == profile_step:
                prof.stop_profile()
                prof.print_model_profile(top_modules=10)
                params = prof.get_total_params()
                flops = prof.get_total_flops()
                macs = prof.get_total_macs()
                prof.end_profile()
                assert within_range(flops, 866076672, TOLERANCE)
                assert within_range(macs, 426516480, TOLERANCE)
                assert params == 185622


if __name__ == "__main__":
    unittest.main()
