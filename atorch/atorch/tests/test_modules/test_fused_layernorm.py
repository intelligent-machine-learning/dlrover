import copy
import json
import unittest
from itertools import product

import pandas as pd
import torch

from atorch.auto.accelerate import model_transform
from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.model_context import ModelContext
from atorch.auto.opt_lib.optimization_library import OptimizationLibrary
from atorch.auto.strategy import Strategy
from atorch.normalization import AtorchLayerNorm
from atorch.normalization import LayerNorm as ApexLayerNorm
from atorch.utils.parse_trace_json import analyze_gpu_kernel, prepare_df


class SimpleLN(torch.nn.Module):
    def __init__(self, shape, layernorm_cls=torch.nn.LayerNorm):
        super().__init__()
        self.ln = layernorm_cls(shape)

    def forward(self, input_tensor):
        return self.ln(input_tensor)


@unittest.skipIf(not torch.cuda.is_available(), "No gpu available for cuda tests")
class FusedLayerNormTest(unittest.TestCase):
    def _run_forward(self, dtype, model, input_tensor, amp_enabled):
        ln_model = model.to("cuda")
        AutoAccelerateContext.counter += 1
        fused_ln_module = ApexLayerNorm(input_tensor.shape[1:]).to("cuda")
        mc = ModelContext(ln_model)
        amp_native_opt = ("amp_native", {"dtype": dtype}, False)
        opt_list = [("module_replace", None, False)]
        if amp_enabled:
            opt_list.append(amp_native_opt)
        strategy = Strategy(opt_list)
        opt_lib = OptimizationLibrary()
        model_context = model_transform(mc, strategy, opt_lib)
        auto_acc_ln_module = model_context.model
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=dtype):
            fused_ln_output = fused_ln_module(input_tensor)
        auto_acc_ln_module_output = auto_acc_ln_module(input_tensor)
        self.assertTrue(torch.allclose(fused_ln_output, auto_acc_ln_module_output))
        del model_context

    def test_fused_layer_norm(self):
        shape = (20, 5, 10, 10)
        model = SimpleLN(shape[1:])
        dtypes = [torch.float32, torch.float16]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            input_tensor = torch.randn(*shape, dtype=dtype, device="cuda")
            self._run_forward(dtype, copy.deepcopy(model), input_tensor, dtype != torch.float32)

    def __triton_layernorm(self, dtype):
        shape = (20, 1024, 1024)
        origin_model = SimpleLN(shape[1:])
        apex_model = SimpleLN(shape[1:], layernorm_cls=ApexLayerNorm)
        apex_model.load_state_dict(origin_model.state_dict())
        atorch_model = SimpleLN(shape[1:], layernorm_cls=AtorchLayerNorm)
        atorch_model.load_state_dict(origin_model.state_dict())
        origin_model.to(dtype)
        apex_model.to(dtype)
        atorch_model.to(dtype)
        input_tensor = torch.randn(*shape, dtype=dtype, device="cuda")

        out_origin = origin_model(input_tensor)
        out_apex = apex_model(input_tensor)
        out_atorch = atorch_model(input_tensor)

        torch.testing.assert_close(out_origin, out_atorch)
        torch.testing.assert_close(out_origin, out_apex)

    @unittest.skipIf(torch.__version__ < (2, 0), "triton need torch 2.0")
    def test_layernorm(self):
        sizes = [
            [4096, 4096],
            [768, 768],
            [512, 768],
            # [6,4],
        ]
        dtypes = [torch.float16, torch.float32]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for (M, N), dtype in product(sizes, dtypes):
            with self.subTest(dtype=dtype, M=M, N=N):
                self.__test_layer_norm(M, N, dtype)

    def __test_layer_norm(self, M, N, dtype, eps=1e-5, device="cuda"):
        # create data
        x_shape = (M * 4, N)
        w_shape = (x_shape[-1],)
        x = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)
        dy = 0.1 * torch.randn_like(x)
        x.requires_grad_(True)
        origin_ln_layer = torch.nn.LayerNorm(w_shape).to(x.device).to(dtype)
        origin_ln_layer.train()
        origin_ln_layer.zero_grad()
        # forward pass
        for layer_cls in [
            ApexLayerNorm,
            AtorchLayerNorm,
        ]:
            atorch_ln_layer = layer_cls(w_shape).to(x.device).to(dtype)
            atorch_ln_layer.load_state_dict(origin_ln_layer.state_dict())
            origin_ln_layer.train()
            origin_ln_layer.zero_grad()

            atorch_ln_layer.train()
            with torch.autocast(device, dtype=dtype):
                y_tri = atorch_ln_layer(x)
                y_ref = origin_ln_layer(x).to(dtype)
                # torch layernorm output is always fp32
                # y_ref = y_ref.to(dtype)
                # backward pass (triton)
                x.grad = None
                atorch_ln_layer.zero_grad()

                y_tri.backward(dy, retain_graph=True)
                dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, atorch_ln_layer.weight, atorch_ln_layer.bias]]
                # backward pass (torch)
                x.grad = None
                origin_ln_layer.zero_grad()
                y_ref.backward(dy, retain_graph=True)
                dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, origin_ln_layer.weight, origin_ln_layer.bias]]

            # compare, if grad is float16, cast to float32 will not lost precision
            def msg_fn(s):
                return "%s, cls=%s dtype=%s" % (s, layer_cls, dtype)

            # compare, if grad is float16, cast to float32 will not lost precision
            with self.subTest("y"):
                torch.testing.assert_close(
                    y_tri.to(torch.float32), y_ref.to(torch.float32), atol=2e-2, rtol=1e-2, msg=msg_fn
                )
            with self.subTest("dx"):
                torch.testing.assert_close(
                    dx_tri.to(torch.float32), dx_ref.to(torch.float32), atol=1e-2, rtol=1e-2, msg=msg_fn
                )
            with self.subTest("db"):
                torch.testing.assert_close(db_tri, db_ref, atol=1e-2, rtol=1e-2, msg=msg_fn)
            with self.subTest("dw"):
                # TODO: fix dw acc
                # torch.testing.assert_close(
                # dw_tri.to(torch.float32), dw_ref.to(torch.float32), atol=2e-2, rtol=1e-2, msg=msg_fn
                # )
                pass

    def get_clear_kernel_df(self, path):
        with open(path) as f:
            traceEvents = json.load(f)["traceEvents"]
        df = pd.DataFrame(traceEvents)
        df["cat"] = df["cat"].str.lower()
        kernel_df = df.query("cat == 'kernel'")
        cat2kernelname = {  # forward kernel/y grad backward/ w+b backward
            "torch": ["vectorized_layer_norm_kernel", "layer_norm_grad_input_kernel", "GammaBetaBackwardCUDAKernel"],
            "atorch": ["_layer_norm_fwd_fused", "_layer_norm_bwd_dx_fused", "_layer_norm_bwd_dwdb", "sum_functor"],
            "apex": [
                "cuApplyLayerNorm",
                "cuComputeGradInput",
                "cuComputePartGradGammaBeta",
            ],
        }
        all_kernelnames = []
        for kernel_list in cat2kernelname.values():
            all_kernelnames.extend(kernel_list)

        def convert_fn(name):
            for kernelname in all_kernelnames:
                if kernelname in name:
                    return kernelname
            else:
                return name

        kernel_df["name"] = kernel_df["name"].apply(convert_fn)
        return kernel_df.query("name ==@all_kernelnames")

    @unittest.skipIf(torch.__version__ < (2, 0), "triton need torch 2.0")
    def test_layernorm_trace(self):
        M, N = 4096, 4096
        x_shape = (M, N)
        w_shape = (x_shape[-1],)
        eps = 1e-5
        device = "cuda"
        dtype = torch.float16
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        x = torch.randn(x_shape, dtype=dtype, device=device, requires_grad=True)
        dy = torch.randn_like(x)
        atorch_ln_layer = AtorchLayerNorm(w_shape).to(x.device).to(x.dtype)
        with torch.no_grad():
            atorch_ln_layer.weight.copy_(weight)
            atorch_ln_layer.bias.copy_(bias)
        atorch_ln_layer.train()
        y_tri = atorch_ln_layer(x)
        apex_layer_norm = ApexLayerNorm(w_shape).to(x.device).to(x.dtype)
        with torch.no_grad():
            apex_layer_norm.weight.copy_(weight)
            apex_layer_norm.bias.copy_(bias)
        # warmup
        for _ in range(10):
            y_tri = atorch_ln_layer(x)
            y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)
            apex_layer_norm(x)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=True,
        ) as prof:
            x.grad, weight.grad, bias.grad = None, None, None
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(10):
                y_tri = atorch_ln_layer(x)
                y_tri.backward(dy, retain_graph=True)
                x.grad, weight.grad, bias.grad = None, None, None
            end_event.record()
            torch.cuda.synchronize()
            atorch_cost = start_event.elapsed_time(end_event)

            x.grad, weight.grad, bias.grad = None, None, None
            start_event.record()
            for _ in range(10):
                y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)
                y_ref.backward(dy, retain_graph=True)
                x.grad, weight.grad, bias.grad = None, None, None
            end_event.record()
            torch.cuda.synchronize()
            native_cost = start_event.elapsed_time(end_event)
            x.grad, weight.grad, bias.grad = None, None, None
            start_event.record()
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(10):
                y_ref = apex_layer_norm(x)
                y_ref.backward(dy, retain_graph=True)
                x.grad, weight.grad, bias.grad = None, None, None
            end_event.record()
            torch.cuda.synchronize()
            # print("apex native",)
            apex_cost = start_event.elapsed_time(end_event)
            print(f"e2e kernel cost: atorch={atorch_cost:.3f} native={native_cost:.3f} apex={apex_cost:.3f}")
            # self.assertLess(atorch_cost, native_cost)
            # self.assertLess(atorch_cost, apex_cost)

        prof.export_chrome_trace("layernorm_trace_%s.json" % dtype)
        kernel_df = self.get_clear_kernel_df("layernorm_trace_%s.json" % dtype)

        cat2kernelname = {  # forward kernel/y grad backward/ w+b backward
            "torch": ["vectorized_layer_norm_kernel", "layer_norm_grad_input_kernel", "GammaBetaBackwardCUDAKernel"],
            "atorch": ["_layer_norm_fwd_fused", "_layer_norm_bwd_dx_fused", "_layer_norm_bwd_dwdb", "sum_functor"],
            "apex": [
                "cuApplyLayerNorm",
                "cuComputeGradInput",
                "cuComputePartGradGammaBeta",
            ],
        }

        cat2times = {key: 0 for key in cat2kernelname}
        for key, kernelnames in cat2kernelname.items():
            for kernelname in kernelnames:
                cat2times[key] += float(
                    kernel_df.query("name == '%s'" % kernelname).groupby("name")["dur"].mean().iloc[0]
                )
        print(
            f"cat2times: atorch={cat2times['atorch']:.3f} native={cat2times['torch']:.3f} apex={cat2times['apex']:.3f}"
        )
        with open("layernorm_trace_%s.json" % dtype, "r") as fin:
            json_obj = json.load(fin)  # TODO: iter json_obj,save memory usage
            df = prepare_df(json_obj)
            ret = analyze_gpu_kernel(df)
            print("compute summary:", ret)
        # self.assertLess(cat2times["atorch"], cat2times["torch"])
        # self.assertLess(cat2times["atorch"], cat2times["apex"])

    @unittest.skipIf(torch.__version__ < (2, 0), "triton need torch 2.0")
    def test_frozen_layer(self):
        device = torch.cuda.current_device()

        class SimpleModule(torch.nn.Module):
            def __init__(self, input_shape, output_shape, layernorm_cls=AtorchLayerNorm):
                super().__init__()
                self.linear1 = torch.nn.Linear(input_shape, input_shape)
                self.layernorm1 = layernorm_cls(input_shape)
                self.linear2 = torch.nn.Linear(input_shape, output_shape)
                self.layernorm2 = layernorm_cls(input_shape)

            def forward(self, x):
                x = self.linear1(x)
                x = self.layernorm1(x)
                x = self.linear2(x)
                x = self.layernorm2(x)

                return x

        B, S, H = 64, 128, 512
        # expect_counts , kernel number:

        # GammaBetaBackwardCUDAKernel
        # _layer_norm_bwd_dwdb
        # _layer_norm_bwd_dx_fused
        # _layer_norm_fwd_fused
        # layer_norm_grad_input_kernel
        # sum_functor
        # vectorized_layer_norm_kernel
        for weight_grad, bias_grad, expect_counts in [
            [
                True,
                True,
                [2, 2, 2, 2, 2, 8, 2],
            ],
            [
                True,
                False,
                [2, 2, 2, 2, 2, 6, 2],
            ],
            [False, True, [2, 1, 2, 2, 2, 8, 2]],
            [False, False, [1, 1, 2, 2, 2, 6, 2]],
        ]:
            # example: when weight_grad=False, bias_grad=False, there are one GammaBetaBackwardCUDAKernel,
            with self.subTest("weight_grad=%s bias_grad=%s" % (weight_grad, bias_grad)):
                model = SimpleModule(H, H)
                model.to(device)
                model.layernorm1.weight.requires_grad_(weight_grad)
                model.layernorm1.bias.requires_grad_(bias_grad)

                model_torch = SimpleModule(H, H, torch.nn.LayerNorm)
                model_torch.to(device)
                model_torch.layernorm1.weight.requires_grad_(weight_grad)
                model_torch.layernorm1.bias.requires_grad_(bias_grad)

                x = torch.randn(B, S, H, device=device, requires_grad=True)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    with_stack=True,
                    with_modules=True,
                ) as prof:
                    y = model(x)
                    dy = torch.randn_like(y)
                    y.backward(dy)
                    model_torch(x).backward(dy)
                prof.export_chrome_trace("layernorm_frozen_trace.json")

                if not weight_grad:
                    self.assertIsNone(model.layernorm1.weight.grad)
                    self.assertIsNone(model_torch.layernorm1.weight.grad)
                else:
                    self.assertIsNotNone(model.layernorm1.weight.grad)
                    self.assertIsNotNone(model_torch.layernorm1.weight.grad)

                if not bias_grad:
                    self.assertIsNone(model.layernorm1.bias.grad)
                    self.assertIsNone(model_torch.layernorm1.bias.grad)
                else:
                    self.assertIsNotNone(model.layernorm1.bias.grad)
                    self.assertIsNotNone(model_torch.layernorm1.bias.grad)
                self.assertIsNotNone(model.linear1.weight.grad)
                self.assertIsNotNone(model_torch.linear1.weight.grad)
                self.assertIsNotNone(model.linear1.bias.grad)
                self.assertIsNotNone(model_torch.linear1.bias.grad)
                kernel_df = self.get_clear_kernel_df("layernorm_frozen_trace.json")
                self.assertListEqual(kernel_df.groupby("name").count().sort_index()["cat"].tolist(), expect_counts)
