import functools
import unittest
from copy import deepcopy

import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.testing._internal.common_utils import TestCase

from atorch.utils.import_util import is_torch_npu_available


class TestOptim(TestCase):
    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    def _build_params_dict(self, weight, bias, **kwargs):
        return [{"params": [weight]}, dict(params=[bias], **kwargs)]

    def _test_basic_cases_template(self, weight, bias, input, constructor, scheduler_constructors):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)
        optimizer = constructor(weight, bias)
        schedulers = []
        for scheduler_constructor in scheduler_constructors:
            schedulers.append(scheduler_constructor(optimizer))

        # to check if the optimizer can be printed as a string
        optimizer.__repr__()

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().item()
        for _i in range(200):
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    val_loss = fn()
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            optimizer.step(fn)
        self.assertLess(fn().item(), initial_value)

    def _test_state_dict(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for _i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        weight_c = Variable(weight.data.clone(), requires_grad=True)
        bias_c = Variable(bias.data.clone(), requires_grad=True)
        optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        # Make sure state_dict_c isn't modified by merely calling load_state_dict
        self.assertEqual(state_dict, state_dict_c)
        # Run both optimizations in parallel
        for _i in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)

        # Make sure state dict is deterministic with equal but not
        # identical parameters
        self.assertEqual(optimizer.state_dict(), optimizer_c.state_dict())
        # Make sure repeated parameters have identical representation
        # in state dict
        optimizer_c.param_groups.extend(optimizer_c.param_groups)
        self.assertEqual(
            optimizer.state_dict()["param_groups"][-1],
            optimizer_c.state_dict()["param_groups"][-1],
        )

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        input_cuda = Variable(input.data.float().cuda())
        weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)
        bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)
        optimizer_cuda = constructor(weight_cuda, bias_cuda)
        fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda, bias_cuda)

        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_c)

        # Make sure state_dict_c isn't modified by merely calling load_state_dict
        self.assertEqual(state_dict, state_dict_c)

        for _i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda)

        # validate deepcopy() copies all public attributes
        def getPublicAttr(obj):
            return set(k for k in obj.__dict__ if not k.startswith("_"))

        self.assertEqual(getPublicAttr(optimizer), getPublicAttr(deepcopy(optimizer)))

    def _test_basic_cases(
        self,
        constructor,
        scheduler_constructors=None,
        ignore_multidevice=False,
    ):
        if scheduler_constructors is None:
            scheduler_constructors = []
        self._test_state_dict(torch.randn(10, 5), torch.randn(10), torch.randn(5), constructor)
        self._test_basic_cases_template(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor,
            scheduler_constructors,
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0],
            torch.randn(5),
            constructor,
            scheduler_constructors,
        )
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(),
            constructor,
            scheduler_constructors,
        )
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0),
            constructor,
            scheduler_constructors,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "no gpu found here.")
    def test_agd(self):
        from atorch.optimizers import AGD

        optimizer = AGD
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, delta=1e-8))
        self._test_basic_cases(lambda weight, bias: optimizer(self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=1))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=1, amsgrad=True))
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            optimizer(None, lr=1e-2, weight_decay=-1)

    @unittest.skipIf(not torch.cuda.is_available(), "no gpu found here.")
    def test_q_adamw(self):
        from atorch.optimizers.low_bit import Q_AdamW

        optimizer = Q_AdamW
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, eps=1e-8))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=1e-2))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, q_bits=8))
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 0"):
            optimizer(None, lr=1e-3, q_bits=0)
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 9"):
            optimizer(None, lr=1e-3, q_bits=9)
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 3.5"):
            optimizer(None, lr=1e-3, q_bits=3.5)

    @unittest.skipIf(not torch.cuda.is_available(), "no gpu found here.")
    def test_q_agd(self):
        from atorch.optimizers.low_bit import Q_AGD

        optimizer = Q_AGD
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, delta=1e-8))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=1e-2))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, amsgrad=True))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, q_bits=8))
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 0"):
            optimizer(None, lr=1e-3, q_bits=0)
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 9"):
            optimizer(None, lr=1e-3, q_bits=9)
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 3.5"):
            optimizer(None, lr=1e-3, q_bits=3.5)

    @unittest.skipIf(not torch.cuda.is_available(), "no gpu found here.")
    def test_q_came(self):
        from atorch.optimizers.low_bit import Q_CAME

        optimizer = Q_CAME
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, eps=(1e-30, 1e-16)))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=1e-2))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, q_bits=8))
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 0"):
            optimizer(None, lr=1e-3, q_bits=0)
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 9"):
            optimizer(None, lr=1e-3, q_bits=9)
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 3.5"):
            optimizer(None, lr=1e-3, q_bits=3.5)

    @unittest.skipIf(not torch.cuda.is_available(), "no gpu found here.")
    def test_q_adafactor(self):
        from atorch.optimizers.low_bit import Q_Adafactor

        optimizer = Q_Adafactor
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer(self._build_params_dict(weight, bias, lr=1e-2), lr=1e-3))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, eps2=(1e-30, 1e-3)))
        self._test_basic_cases(lambda weight, bias: optimizer([weight, bias], lr=1e-3, beta1=0.9))
        self._test_basic_cases(
            lambda weight, bias: optimizer(
                [weight, bias], lr=1e-3, weight_decay=1e-2, scale_parameter=False, relative_step=False
            )
        )
        self._test_basic_cases(
            lambda weight, bias: optimizer(
                [weight, bias], lr=1e-3, scale_parameter=False, relative_step=False, q_bits=8
            )
        )
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 0"):
            optimizer(None, lr=1e-3, q_bits=0)
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 9"):
            optimizer(None, lr=1e-3, q_bits=9)
        with self.assertRaisesRegex(ValueError, "Invalid q_bits value: 3.5"):
            optimizer(None, lr=1e-3, q_bits=3.5)

    @unittest.skipIf(not is_torch_npu_available(), "no npu found here")
    def test_npu_adamw(self):
        from atorch.npu.optim import NpuAdamW

        optimizer = NpuAdamW
        self._test_basic_cases_template(
            torch.randn(10, 5).npu(),
            torch.randn(10).npu(),
            torch.randn(5).npu(),
            lambda weight, bias: optimizer([weight, bias], lr=1e-3),
            [],
        )
        self._test_basic_cases_template(
            torch.randn(10, 5).npu(),
            torch.randn(10).npu(),
            torch.randn(5).npu(),
            lambda weight, bias: optimizer(self._build_params_dict(weight, bias, lr=1e-3), lr=1e-3),
            [],
        )
        self._test_basic_cases_template(
            torch.randn(10, 5).npu(),
            torch.randn(10).npu(),
            torch.randn(5).npu(),
            lambda weight, bias: optimizer([weight, bias], lr=1e-3, eps=1e-8),
            [],
        )
        self._test_basic_cases_template(
            torch.randn(10, 5).npu(),
            torch.randn(10).npu(),
            torch.randn(5).npu(),
            lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=1e-2),
            [],
        )
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
            optimizer(None, lr=1e-2, weight_decay=-1)

    @unittest.skipIf(not is_torch_npu_available(), "no npu found here")
    def test_npu_adamw_compare_with_cpu(self):
        from atorch.npu.optim import NpuAdamW

        weight_value = torch.randn(10, 5)
        bias_value = torch.randn(10)
        input_value = torch.randn(5)
        weight_cpu = Variable(weight_value, requires_grad=True)
        bias_cpu = Variable(bias_value, requires_grad=True)
        input_cpu = Variable(input_value)

        weight_npu = Variable(weight_value.npu(), requires_grad=True)
        bias_npu = Variable(bias_value.npu(), requires_grad=True)
        input_npu = Variable(input_value.npu(), requires_grad=True)

        ori_optim = torch.optim.AdamW([weight_cpu, bias_cpu], lr=1e-3)
        ori_optim.zero_grad()
        y_cpu = weight_cpu.mv(input_cpu)
        loss_cpu = (y_cpu + bias_cpu).pow(2).sum()
        loss_cpu.backward()
        ori_optim.step()

        npu_optim = NpuAdamW([weight_npu, bias_npu], lr=1e-3)
        npu_optim.zero_grad()
        y_npu = weight_npu.mv(input_npu)
        loss_npu = (y_npu + bias_npu).pow(2).sum()
        loss_npu.backward()
        npu_optim.step()

        torch.testing.assert_close(weight_npu.cpu(), weight_cpu)
        torch.testing.assert_close(bias_npu.cpu(), bias_cpu)


class TestSamOptim(TestCase):
    def setUp(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super(TinyModel, self).__init__()
                self.linear = torch.nn.Linear(5, 10)

            def forward(self, x):
                x = self.linear(x)
                return torch.norm(x)

        self._model_class = TinyModel

    def _test_basic_case_template(self, input, model, optimizer):
        if torch.cuda.is_available():
            input = input.cuda()
            model = model.cuda()

        def closure():
            loss = model(input)
            loss.backward()
            return loss

        initial_value = closure().item()
        for _ in range(100):
            loss = optimizer.step(closure)
            optimizer.zero_grad()

        self.assertLess(loss.item(), initial_value)

    def _test_basic_cases(self, constructor, **args):
        model = self._model_class()
        base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
        self._test_basic_case_template(
            input=torch.randn(5),
            model=model,
            optimizer=constructor(model, base_opt, **args),
        )
        model = self._model_class()
        base_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        self._test_basic_case_template(
            input=torch.randn(5),
            model=model,
            optimizer=constructor(model, base_opt, **args),
        )

    @unittest.skipIf(not torch.cuda.is_available(), "no gpu found here.")
    def test_wsam(self):
        from atorch.optimizers import WeightedSAM

        self._test_basic_cases(WeightedSAM, rho=0.1)
        self._test_basic_cases(WeightedSAM, rho=0.1, gamma=0.5)
        with self.assertRaisesRegex(AssertionError, "Invalid rho, should be non-negative: -1"):
            WeightedSAM(None, None, rho=-1)


if __name__ == "__main__":
    unittest.main()
