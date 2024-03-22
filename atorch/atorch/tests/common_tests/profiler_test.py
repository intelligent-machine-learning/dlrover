import copy
import unittest
from contextlib import contextmanager

import torch

from atorch.normalization import LayerNorm
from atorch.utils import AProfiler
from atorch.utils.prof import GlobalContext, _patch_functionals, _reload_functionals


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


@contextmanager
def patch_context():
    origin_module_flop_count = copy.deepcopy(GlobalContext.module_flop_count)
    try:
        GlobalContext.module_flop_count.append([])
        _patch_functionals()
        yield GlobalContext.module_flop_count
    finally:
        GlobalContext.module_flop_count = origin_module_flop_count
        _reload_functionals()


class TestProfiler(unittest.TestCase):
    def test_softmax_compute(self):
        from atorch.utils.prof import _softmax_flops_compute

        input = torch.randn(2, 1, 32, 32, device=device)
        flops, mac = _softmax_flops_compute(input)
        self.assertEqual(flops, 2097152)

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
                prof.compute_gpu_utilization(10)
                prof.end_profile()
                assert within_range(flops, 866076672, TOLERANCE)
                assert within_range(macs, 426516480, TOLERANCE)
                assert params == 185622

    @unittest.skipIf(
        torch.__version__ < (2, 0),
        "sdp need torch 2.0",
    )
    def test_patch_functional(self):
        batch_size, nheads, seqlen, headdim = 2, 8, 32, 64
        from torch.nn import LayerNorm

        torch_layer = LayerNorm((seqlen, nheads, headdim))
        with patch_context():
            q = k = v = torch.randn(batch_size, seqlen, nheads, headdim)
            torch.nn.functional.scaled_dot_product_attention(q, k, v)

            torch_layer(q)
            self.assertListEqual(
                GlobalContext.module_flop_count[-1], [("scaled_dot_product_attention", 2555968), ("layer_norm", 163840)]
            )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.__version__ < (2, 0),
        "No gpu available for cuda tests",
    )
    def test_patch_functional_gpu2(self):
        batch_size, nheads, seqlen, headdim = 2, 8, 32, 64
        from atorch.normalization import AtorchLayerNorm

        atorch_layer = AtorchLayerNorm((seqlen, nheads, headdim))
        atorch_layer = atorch_layer.to("cuda")
        with patch_context():
            q = k = v = torch.randn(batch_size, seqlen, nheads, headdim, device="cuda")
            torch.nn.functional.scaled_dot_product_attention(q, k, v)

            atorch_layer(q)
            print("GlobalContext.module_flop_count", GlobalContext.module_flop_count)
            self.assertListEqual(
                GlobalContext.module_flop_count[-1],
                [("scaled_dot_product_attention", 2555968), ("AtorchLayerNormFunc", 163840)],
            )


if __name__ == "__main__":
    unittest.main()
