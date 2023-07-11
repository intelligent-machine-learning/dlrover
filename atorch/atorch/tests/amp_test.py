import logging
import sys
import unittest
from unittest.mock import patch

import torch

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.amp_optimization import (
    AmpApexO1Optimization,
    AmpApexO2Optimization,
    AmpNativeOptimization,
    AmpNativeOptimizer,
)
from atorch.auto.opt_lib.half_optimization import HalfOptimization
from atorch.tests.test_utils import skip_if_old_gpu
from atorch.tests.toy_module import create_model_context, run_train


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
        skip_if_old_gpu(),
        "Skip gpu older than a100.",
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
        skip_if_old_gpu(),
        "Skip gpu older than a100.",
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


if __name__ == "__main__":
    unittest.main()
