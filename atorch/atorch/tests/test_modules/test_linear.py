import os
import unittest

import torch
from torch import nn

from atorch.modules.transformer.linear import AmpFusedDense, Linear2D, replace_linear


class MyModule(torch.nn.Module):
    def __init__(self, use_origin_layer, input_dim, output_dim):
        super().__init__()
        if use_origin_layer:
            linear_cls = torch.nn.Linear
        else:
            linear_cls = Linear2D
        self.fc1 = linear_cls(input_dim, output_dim)
        self.fc2 = linear_cls(output_dim, input_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


@unittest.skipIf(True, "Failed on gpu")
class TestsLossScaler(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1241)
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return super().setUp()

    def test_amp_module(self):
        device = torch.cuda.current_device()
        batch, seq_length = 10, 500
        input_dim, output_dim = 128, 666
        linear0 = torch.nn.Linear(input_dim, output_dim)
        x = torch.randn(batch, seq_length, input_dim, device=device)
        mod = AmpFusedDense(input_dim, output_dim)
        mod.to(device)
        linear0.to(device)
        linear0.zero_grad()
        mod.weight.data.copy_(linear0.weight)
        mod.bias.data.copy_(linear0.bias)
        out = mod(x.reshape(batch * seq_length, -1))
        out = out.reshape(batch, seq_length, -1)
        out1 = linear0(x)
        torch.testing.assert_close(out, out1)
        Y = torch.randn_like(out)
        loss = torch.sum(torch.pow(Y - out, 2))
        loss.backward()
        loss = torch.sum(torch.pow(Y - out1, 2))
        loss.backward()
        for (name, p0), p1 in zip(mod.named_parameters(), linear0.parameters()):
            if p0.grad is not None:
                with self.subTest("grad check", name=name):
                    torch.testing.assert_allclose(p0.grad, p1.grad)


class TestLinear(unittest.TestCase):
    def test_2d_linear(self):
        x = torch.randn(2, 3, 4)
        linear = Linear2D(4, 5)
        old_linear = nn.Linear(4, 5)
        old_linear.load_state_dict(linear.state_dict())

        y = linear(x)
        self.assertEqual(y.size(), (2, 3, 5))

        y1 = old_linear(x)
        self.assertEqual(y1.size(), (2, 3, 5))
        torch.testing.assert_close(y, y1)
        Y = torch.randn_like(y)
        loss = torch.nn.functional.mse_loss(y, Y)
        loss1 = torch.nn.functional.mse_loss(y1, Y)
        loss.backward()
        loss1.backward()
        name2grad = {}
        for name, p in linear.named_parameters():
            name2grad[name] = p.grad
        for name, p in old_linear.named_parameters():
            torch.testing.assert_close(name2grad[name], p.grad, rtol=1e-4, atol=1e-4)

    @unittest.skipIf(not torch.cuda.is_available(), "this case run on gpu")
    def test_2d_linear_gpu(self):
        device = torch.device("cuda")
        dtype = torch.float32
        batch, head_dim, head_num, hidden = 128, 64, 8, 768
        x = torch.randn(batch, head_dim, head_num, hidden, device=device, dtype=dtype)
        dst_dim = 1024
        linear = Linear2D(hidden, dst_dim)
        linear.to(device).to(dtype)
        old_linear = nn.Linear(hidden, dst_dim)

        old_linear.load_state_dict(linear.state_dict())
        old_linear.to(device).to(dtype)
        for i in range(5):  # warmup
            y = linear(x)
            y1 = old_linear(x)
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        start_event2 = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        end_event2 = torch.cuda.Event(enable_timing=True)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        ) as prof:
            start_event.record()
            for i in range(100):
                y = linear(x)
            end_event.record()
            torch.cuda.synchronize()
            fused_time = start_event.elapsed_time(end_event)

            start_event2.record()
            for i in range(100):
                y1 = old_linear(x)
            end_event2.record()

            torch.cuda.synchronize()
            origin_time = start_event2.elapsed_time(end_event2)
            torch.cuda.synchronize()
            Y = torch.randn_like(y)
            loss = torch.nn.functional.mse_loss(y, Y)
            loss1 = torch.nn.functional.mse_loss(y1, Y)
            loss.backward()
            loss1.backward()
            torch.testing.assert_close(y, y1, rtol=1e-4, atol=1e-4)
            name2grad = {}
            for name, p in linear.named_parameters():
                name2grad[name] = p.grad
            for name, p in old_linear.named_parameters():
                torch.testing.assert_close(name2grad[name], p.grad, rtol=1e-4, atol=1e-4)
            print("fused_time", fused_time, "origin_time", origin_time)
        prof.export_chrome_trace("test_2d_linear_gpu.json")

    @unittest.skipIf(not torch.cuda.is_available(), "this case run on gpu")
    def test_two_layer(self):
        device = torch.device("cuda")
        dtype = torch.float32
        batch, head_dim, head_num, hidden = 128, 64, 8, 768
        x = torch.randn(batch, head_dim, head_num, hidden, device=device, dtype=dtype)
        layer = MyModule(True, hidden, hidden).to(device)
        layer.reset_parameters()
        layer_fused = MyModule(False, hidden, hidden).to(device)
        layer_fused.load_state_dict(layer.state_dict())
        with torch.cuda.amp.autocast(dtype=dtype):
            y = layer(x)
            y1 = layer_fused(x)
            torch.testing.assert_close(y, y1, rtol=1e-4, atol=1e-4)
            Y = torch.randn_like(y)
            loss = torch.nn.functional.mse_loss(y, Y)
            loss1 = torch.nn.functional.mse_loss(y1, Y)
            loss.backward()
            loss1.backward()
            name2grad = {}
            for name, p in layer.named_parameters():
                name2grad[name] = p.grad
            for name, p in layer_fused.named_parameters():
                torch.testing.assert_close(name2grad[name], p.grad, rtol=1e-4, atol=1e-4)

    @unittest.skipIf(not torch.cuda.is_available(), "this case run on gpu")
    def test_replace_linear(self):
        device = torch.device("cuda")
        dtype = torch.float32
        batch, head_dim, head_num, hidden = 128, 64, 8, 768
        x = torch.randn(batch, head_dim, head_num, hidden, device=device, dtype=dtype)
        layer = MyModule(True, hidden, hidden)
        layer.reset_parameters()
        replace_linear(layer, "")
        layer.to(device)
        self.assertTrue(isinstance(layer.fc1, Linear2D))
        layer_fused = MyModule(True, hidden, hidden).to(device)
        layer_fused.load_state_dict(layer.state_dict())
        with torch.cuda.amp.autocast(dtype=dtype):
            y = layer(x)
            y1 = layer_fused(x)
            torch.testing.assert_close(y, y1, rtol=1e-4, atol=1e-4)
            Y = torch.randn_like(y)
            loss = torch.nn.functional.mse_loss(y, Y)
            loss1 = torch.nn.functional.mse_loss(y1, Y)
            loss.backward()
            loss1.backward()
            name2grad = {}
            for name, p in layer.named_parameters():
                name2grad[name] = p.grad
            for name, p in layer_fused.named_parameters():
                torch.testing.assert_close(name2grad[name], p.grad, rtol=1e-4, atol=1e-4)
