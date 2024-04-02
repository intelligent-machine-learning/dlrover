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
from atorch.auto.opt_lib.amp_optimization import AmpNativeOptimization, AmpNativeOptimizer
from atorch.auto.opt_lib.half_optimization import HalfOptimization
from atorch.auto.opt_lib.optimization_library import OptimizationLibrary
from atorch.auto.strategy import Strategy
from atorch.common.util_func import find_free_port
from atorch.tests.toy_modules.toy_module import create_model_context, optim_func, run_train
from atorch.tests.utils.test_utils import DummyProcessGroup
from atorch.utils.grad_scaler import BF16GradScaler, BF16ShardedGradScaler


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
        self.assertTrue(optimizer.step_was_skipped)

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
        self.assertFalse(hasattr(optimizer, "step_was_skipped"))


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
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
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
    param_value = torch.tensor(10.0, dtype=torch.float32)
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
    for i in range(2):
        optimizer.zero_grad()
        # step0, rank0 will cause inf. Other steps and ranks will not.
        if i == 0:
            tensor_value = bf16_max if rank == 0 else 1
        else:
            tensor_value = 1
        x = torch.tensor([[tensor_value], [tensor_value]], dtype=torch.bfloat16, device=device)
        y = model(x)
        loss = loss_func(y, None)
        scaler = optimizer.grad_scaler
        if i == 0:
            if rank == 0:
                assert torch.all(loss.isinf())
            else:
                assert not loss.isinf().any().item()
            loss.backward()
            for param in model.module.parameters():
                assert torch.all(param.grad.isnan())
            optimizer.step()
            for param in model.module.parameters():
                assert not torch.all(param.isnan())
            assert scaler.has_overflow()
            assert optimizer.step_was_skipped
        else:
            assert not loss.isinf().any().item()
            loss.backward()
            optimizer.step()
            assert not scaler.has_overflow()
            assert not optimizer.step_was_skipped
        assert scaler._init_scale == 1.0
        assert scaler._growth_factor == 1.0
        assert scaler._backoff_factor == 1.0

    dist.destroy_process_group()


@unittest.skipIf(
    not torch.cuda.is_available(),
    "No gpu available for cuda tests",
)
class BF16GradScalerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.randn(4, 4).cuda()
        self.m = torch.nn.Linear(4, 1).cuda()
        kwargs = {"lr": 0.1}
        self.o = torch.optim.SGD(self.m.parameters(), **kwargs)
        return super().setUp()

    def tearDown(self) -> None:
        del self.x
        del self.m
        del self.o
        return super().tearDown()

    def test_bf16_scaler_has_no_overflow(self):
        scaler = BF16GradScaler()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            y = self.m(self.x)
            loss = y.mean()
        scaler.scale(loss).backward()
        scaler.step(self.o)
        scaler.update()
        self.assertFalse(scaler.has_overflow())
        self.assertEqual(scaler._init_scale, 1.0)
        self.assertEqual(scaler._backoff_factor, 1.0)
        self.assertEqual(scaler._growth_factor, 1.0)

    def test_bf16_scaler_has_overflow(self):
        scaler = BF16GradScaler()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            y = self.m(self.x)
            loss = y.mean()
        scaler.scale(loss).backward()
        with torch.no_grad():
            self.m.weight.grad.fill_(float("NaN"))
        scaler.step(self.o)
        scaler.update()
        self.assertTrue(scaler.has_overflow())
        self.assertEqual(scaler._init_scale, 1.0)
        self.assertEqual(scaler._backoff_factor, 1.0)
        self.assertEqual(scaler._growth_factor, 1.0)

    def test_bf16_scaler_two_steps(self):
        # The first step has overflow and the 2nd does not.
        scaler = BF16GradScaler()
        for i in range(2):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                y = self.m(self.x)
                loss = y.mean()
            scaler.scale(loss).backward()
            if i == 0:
                with torch.no_grad():
                    self.m.weight.grad.fill_(float("NaN"))
            scaler.step(self.o)
            scaler.update()
            self.o.zero_grad()
            if i == 0:
                self.assertTrue(scaler.has_overflow())
            elif i == 1:
                self.assertFalse(scaler.has_overflow())

    def test_bf16_sharded_scaler_has_overflow(self):
        pg = DummyProcessGroup(rank=0, size=1)
        scaler = BF16ShardedGradScaler(process_group=pg)
        loss = torch.full((1,), 4.0, dtype=torch.float32, device="cpu")
        t0 = torch.tensor([float("inf")], dtype=torch.float32, device="cpu")
        t0.grad = t0.clone()
        opt = torch.optim.SGD([t0], lr=1.0)
        scaler.scale(loss)
        scaler.step(opt)
        self.assertTrue(scaler.has_overflow())


if __name__ == "__main__":
    unittest.main()
