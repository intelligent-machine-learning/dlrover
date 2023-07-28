import logging
import os
import sys
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from atorch.auto.accelerate import model_transform
from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.model_context import ModelContext
from atorch.auto.opt_lib.amp_optimization import (
    AmpApexO1Optimization,
    AmpApexO2Optimization,
    AmpNativeOptimization,
    AmpNativeOptimizer,
)
from atorch.auto.opt_lib.half_optimization import HalfOptimization
from atorch.auto.opt_lib.optimization_library import OptimizationLibrary
from atorch.auto.strategy import Strategy
from atorch.tests.toy_module import create_model_context, optim_func, run_train


class AmpOptimizationTest(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_amp_native_scaler(self):
        AutoAccelerateContext.counter += 1
        model_context1 = create_model_context(data_size=4, batch_size=1)
        amp_native_opt = AmpNativeOptimization()
        model_context1 = amp_native_opt.transform(model_context1)
        model_context1.update_dataloader()
        model_context1.update_optim()
        config = {"enabled": True, "dtype": torch.float16}
        amp_native_opt.apply_wrapper(model_context1, "amp_native", wrapper_config=config)
        model_context1.model.cuda()

        config.pop("counter")

        AutoAccelerateContext.counter += 1
        model_context2 = create_model_context(data_size=4, batch_size=1)
        amp_native_opt2 = AmpNativeOptimization()
        model_context2 = amp_native_opt2.transform(model_context2)
        model_context2.update_dataloader()
        model_context2.update_optim()
        amp_native_opt2.apply_wrapper(model_context2, "amp_native", wrapper_config=config)
        model_context2.model.cuda()

        device = "cuda:0"
        run_train(
            model_context1.model,
            dataloader=model_context1.dataloader,
            optim=model_context1.optim,
            prepare_input=model_context1.prepare_input,
            loss_func=model_context1.loss_func,
            device=device,
        )

        run_train(
            model_context2.model,
            dataloader=model_context2.dataloader,
            optim=model_context2.optim,
            prepare_input=model_context2.prepare_input,
            loss_func=model_context2.loss_func,
            device=device,
        )

        scaler1 = model_context1.optim.grad_scaler
        scaler2 = model_context2.optim.grad_scaler
        self.assertNotEqual(id(scaler1), id(scaler2))

        AutoAccelerateContext.reset()

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_amp_apex_optimizer(self):
        AutoAccelerateContext.counter = 1
        model_context1 = create_model_context(data_size=4, batch_size=1)
        amp_apex_opt = AmpApexO1Optimization()
        model_context1 = amp_apex_opt.transform(model_context1)
        model_context1.update_dataloader()
        model_context1.update_optim()
        o1_config = {"enabled": True, "opt_level": "O1"}
        amp_apex_opt.apply_wrapper(model_context1, "amp_apex_o1", wrapper_config=o1_config)
        model_context1.model.cuda()

        device = "cuda:0"

        run_train(
            model_context1.model,
            dataloader=model_context1.dataloader,
            optim=model_context1.optim,
            prepare_input=model_context1.prepare_input,
            loss_func=model_context1.loss_func,
            device=device,
        )

        AutoAccelerateContext.counter += 1
        model_context2 = create_model_context(data_size=4, batch_size=1)
        amp_apex_opt2 = AmpApexO2Optimization()

        with self.assertRaises(RuntimeError):
            model_context2 = amp_apex_opt2.transform(model_context2)

        amp_apex_opt.reset(None)
        amp_apex_opt2.reset(None)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_fp16_and_bf16(self):
        """fp16 needs loss scaling but bf16 does not."""
        AutoAccelerateContext.counter += 1
        model_context1 = create_model_context(data_size=4, batch_size=1)
        fp16_amp_native_opt = AmpNativeOptimization()
        fp16_config = {"dtype": "fp16"}
        model_context1 = fp16_amp_native_opt.transform(model_context1, config=fp16_config)
        model_context1.update_dataloader()
        model_context1.update_optim()
        fp16_amp_native_opt.apply_wrapper(model_context1, "amp_native", wrapper_config=fp16_config)
        self.assertIsInstance(model_context1.optim, AmpNativeOptimizer)

        AutoAccelerateContext.counter += 1
        model_context2 = create_model_context(data_size=4, batch_size=1)
        bf16_amp_native_opt = AmpNativeOptimization()
        bf16_config = {"dtype": "bf16"}
        model_context2 = bf16_amp_native_opt.transform(model_context2, config=bf16_config)
        model_context2.update_dataloader()
        model_context2.update_optim()
        bf16_amp_native_opt.apply_wrapper(model_context2, "amp_native", wrapper_config=bf16_config)
        self.assertIsInstance(model_context2.optim, torch.optim.Adam)

    def test_half(self):
        model_context1 = create_model_context(data_size=4, batch_size=1)
        opt = HalfOptimization()
        opt.tune(model_context1, config="fp16")

        def reraise_logging_exception(record):
            t, v, tb = sys.exc_info()
            raise t(v).with_traceback(tb)

        # let logger raise exception
        with patch.object(logging.Handler, "handleError", side_effect=reraise_logging_exception):
            opt.apply_wrapper(model_context1, wrapper_name="half", wrapper_config="tf32")

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_half_precision_dtype_gpu(self):
        device = "cuda"
        for dtype in [torch.bfloat16, torch.float16]:
            AutoAccelerateContext.counter += 1
            model_context = create_model_context(data_size=4, batch_size=1)
            model_context.model.to(device)
            model_context.update_dataloader()
            model_context.update_optim()
            amp_config = {"dtype": dtype}
            AmpNativeOptimization.apply_wrapper(model_context, "amp_native", amp_config)
            model = model_context.model
            dataloader = model_context.dataloader
            loss_func = model_context.loss_func
            optimizer = model_context.optim
            prepare_input = model_context.prepare_input

            for _, batch in enumerate(dataloader):
                batch = prepare_input(batch, device)
                outputs = model(batch)
                self.assertEqual(outputs.dtype, dtype)
                loss = loss_func(batch, outputs)
                self.assertEqual(loss.dtype, torch.float32)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


class OneVarModule(torch.nn.Module):
    def __init__(self, param_value):
        super().__init__()
        self.var = torch.nn.parameter.Parameter(param_value)

    def forward(self):
        return torch.pow(self.var, 2)


class NonFiniteChecking(unittest.TestCase):
    def setUp(self):
        super().setUp()

        def identity_loss_func(a, _):
            return a

        bf16_max = torch.finfo(torch.bfloat16).max
        self.param_value = torch.tensor(bf16_max, dtype=torch.float32)
        model = OneVarModule(self.param_value)
        self.model_context = ModelContext(
            model=model,
            optim_func=optim_func,
            dataset=None,
            loss_func=identity_loss_func,
            prepare_input=None,
            optim_args={"lr": 1},
        )

    def tearDown(self):
        del self.model_context
        return super().tearDown()

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_skipping_update_params_nonfinite_loss(self):
        bf16_config = {"dtype": "bf16", "skip_if_nonfinite": True}
        bf16_amp_native_opt = AmpNativeOptimization()
        model_context = bf16_amp_native_opt.transform(self.model_context, config=bf16_config)
        model_context.update_optim()
        bf16_amp_native_opt.apply_wrapper(model_context, "amp_native", wrapper_config=bf16_config)
        model = model_context.model.cuda()
        optimizer = model_context.optim
        loss_func = model_context.loss_func
        optimizer.zero_grad()
        y = model()
        loss = loss_func(y, None)
        self.assertTrue(torch.all(loss.isinf()))
        loss.backward()
        self.assertTrue(torch.all(model.var.grad.isinf()))
        optimizer.step()
        self.assertTrue(torch.allclose(model.var, self.param_value))
        scaler = optimizer.grad_scaler
        self.assertEqual(scaler._init_scale, 1.0)
        self.assertEqual(scaler._growth_factor, 1.0)
        self.assertEqual(scaler._backoff_factor, 1.0)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "No gpu available for cuda tests",
    )
    def test_not_skipping_update_params_nonfinite_loss(self):
        bf16_config = {"dtype": "bf16", "skip_if_nonfinite": False}
        bf16_amp_native_opt = AmpNativeOptimization()
        model_context = bf16_amp_native_opt.transform(self.model_context, config=bf16_config)
        model_context.update_optim()
        bf16_amp_native_opt.apply_wrapper(model_context, "amp_native", wrapper_config=bf16_config)
        model = model_context.model.cuda()
        optimizer = model_context.optim
        loss_func = model_context.loss_func
        optimizer.zero_grad()
        y = model()
        loss = loss_func(y, None)
        self.assertTrue(torch.all(loss.isinf()))
        loss.backward()
        self.assertTrue(torch.all(model.var.grad.isinf()))
        optimizer.step()
        self.assertTrue(torch.all(model.var.isnan()))
        self.assertFalse(hasattr(optimizer, "grad_scaler"))


class OneLinearModule(torch.nn.Module):
    def __init__(self, param_value):
        super().__init__()
        self.var = torch.nn.Linear(1, 2)
        for param in self.var.parameters():
            param.detach().fill_(param_value)

    def forward(self, x):
        return self.var(x)


class FSDPNonFiniteChecking(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        return super().tearDown()

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "No gpu available for cuda tests",
    )
    def test_skipping_update_params_nonfinite_loss(self):
        world_size = 2
        mp.spawn(
            _test_skipping_update_params_nonfinite_loss,
            args=(world_size,),
            nprocs=world_size,
            join=True,
            daemon=False,
            start_method="spawn",
        )


def _test_skipping_update_params_nonfinite_loss(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"
    strategy = Strategy(
        [
            ("parallel_mode", ([("data", torch.distributed.get_world_size())], None), False),
            ("fsdp", {"atorch_size_based_min_num_params": 1}, False),
            ("amp_native", {"dtype": torch.bfloat16, "skip_if_nonfinite": True}, False),
        ]
    )
    opt_lib = OptimizationLibrary()

    def norm_loss_func(a, _):
        return a.norm()

    bf16_max = torch.finfo(torch.bfloat16).max
    param_value = torch.tensor(bf16_max, dtype=torch.float32)
    model = OneLinearModule(param_value)
    model_context = ModelContext(
        model=model,
        optim_func=optim_func,
        dataset=None,
        loss_func=norm_loss_func,
        prepare_input=None,
        optim_args={"lr": 1},
    )
    model_context = model_transform(model_context, strategy, opt_lib, create_dataloader=False)
    model = model_context.model
    optimizer = model_context.optim
    loss_func = model_context.loss_func
    optimizer.zero_grad()
    x = torch.tensor([[bf16_max], [bf16_max]], dtype=torch.bfloat16, device=device)
    y = model(x)
    loss = loss_func(y, None)
    assert torch.all(loss.isinf())
    loss.backward()
    for param in model.module.parameters():
        assert torch.all(param.grad.isnan())
    optimizer.step()
    for param in model.module.parameters():
        assert not torch.all(param.isnan())
    scaler = optimizer.grad_scaler
    assert scaler._init_scale == 1.0
    assert scaler._growth_factor == 1.0
    assert scaler._backoff_factor == 1.0


if __name__ == "__main__":
    unittest.main()
