import random
import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase

import atorch


class TestNNDeviceType(TestCase):
    def _test_LayerNorm_general(self, device, dtype=torch.float):
        for i in range(2, 6):
            shape = torch.randint(3, 6, (i,), dtype=torch.long).tolist()
            x = torch.empty(*shape, device=device, dtype=dtype).uniform_(0, 10)
            normalized_ndim = random.randint(1, i - 1)  # inclusive
            normalized_shape = shape[-normalized_ndim:]
            unnormalized_shape = shape[:-normalized_ndim]

            # test that LN normalizes to mean 0 and stddev 1
            ln = atorch.normalization.LayerNorm(normalized_shape, eps=0).to(device, dtype)
            ln.weight.data.fill_(1)
            ln.bias.data.fill_(0)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)

            delta = 1e-1 if dtype == torch.bfloat16 else 1e-5
            self.assertEqual(torch.abs(mean.data).mean(), 0, atol=delta, rtol=0)
            self.assertEqual(torch.abs(var.data).mean(), 1, atol=delta, rtol=0)

            # test that LN applies weight and bias correctly
            scale, bias = torch.empty(2).uniform_(0.2, 2).tolist()
            ln.weight.data.fill_(scale)
            ln.bias.data.fill_(bias)
            output = ln(x)
            out_reshaped = output.view(*(unnormalized_shape + [-1]))
            mean = out_reshaped.mean(-1)
            var = out_reshaped.var(-1, unbiased=False)
            self.assertEqual(torch.abs(mean.data).mean(), bias, atol=delta, rtol=0)
            self.assertEqual(torch.abs(var.data).mean(), scale**2, atol=delta, rtol=0)

        bad_norm_shape_input_shape = {
            (): (),
            (2, 3): (3,),
            (2,): (1, 2, 3),
            (10,): (2, 3),
            10: (2, 3),
        }
        for norm_shape, input_shape in bad_norm_shape_input_shape.items():
            ln = atorch.normalization.LayerNorm(norm_shape)
            input = torch.empty(input_shape, device=device, dtype=dtype).uniform_(0, 10)
            self.assertRaises(RuntimeError, lambda: ln(input))

    def _test_LayerNorm_cuda_half(self, device):
        input = torch.empty(2, 3, 3, 2, device=device, dtype=torch.half).random_(1, 10).requires_grad_(True)
        m = atorch.normalization.LayerNorm([3, 2]).to(device, torch.half)
        output = m(input)
        output.sum().backward()
        self.assertEqualTypeString(output, input)

    @unittest.skipIf(True, "Failed on gpu")
    def test_LayerNorm_general(self, device):
        self._test_LayerNorm_general(device)

        # TODO(ya): BFloat16
        # if self.device_type == 'cuda' or self.device_type == 'cpu':
        #    self._test_LayerNorm_general(device, dtype=torch.bfloat16)

        if self.device_type == "cuda":
            self._test_LayerNorm_cuda_half(device)

    @unittest.skipIf(True, "Failed on gpu")
    def test_LayerNorm_numeric(self, device):
        def layer_norm_ref(X, gamma, beta, normalized_shape, eps):
            feature_size = torch.prod(torch.tensor(normalized_shape))
            X_view = X.view(-1, feature_size)
            mean = X_view.mean(dim=-1, keepdim=True)
            var = X_view.var(dim=-1, unbiased=False, keepdim=True)
            Y = (X_view - mean) / torch.sqrt(var + eps)
            Y = Y * gamma.view(-1) + beta.view(-1)
            return Y.view(*X.size())

        normalized_shape = [256, 256, 144]
        layer_norm = atorch.normalization.LayerNorm(normalized_shape).float().to(device)
        # layer_norm = nn.LayerNorm(normalized_shape).float().to(device)
        X = torch.rand(2, *normalized_shape, dtype=torch.float32, device=device)

        Y = layer_norm(X)
        Y_ref = layer_norm_ref(
            X,
            layer_norm.weight.data,
            layer_norm.bias.data,
            normalized_shape,
            layer_norm.eps,
        )
        self.assertEqual(Y, Y_ref, rtol=0, atol=1e-2)

        if self.device_type == "cuda":
            layer_norm.cpu()
            Y_cpu = layer_norm(X.cpu())
            self.assertEqual(Y_cpu, Y, rtol=0, atol=1e-2)


instantiate_device_type_tests(TestNNDeviceType, globals())

if __name__ == "__main__":
    unittest.main()
