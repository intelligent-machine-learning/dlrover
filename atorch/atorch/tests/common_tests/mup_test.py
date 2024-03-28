import math
import unittest

import torch
from torch import nn
from torch.nn import Linear

from atorch.mup.infshape import InfDim, InfShape, zip_infshape
from atorch.mup.module import MupLinear, MupModule, OutputLayer, QKVLayer, QLayer
from atorch.mup.optim import MuAdam, MuSGD
from atorch.mup.shape import get_infshapes, get_shapes, make_base_shapes, save_base_shapes, set_base_shapes


def _init_model(model, std=0.02):
    for param in model.parameters():
        if len(param.shape) >= 2:
            nn.init.normal_(param, std=std)
    return model


def _generate_MLP(width, bias=True, batchnorm=False, device="cpu"):
    mods = [
        Linear(1024, width, bias=bias, device=device),
        nn.ReLU(),
        Linear(width, width, bias=bias, device=device),
        nn.ReLU(),
        Linear(width, 10, bias=bias, device=device),
    ]
    if batchnorm:
        mods.insert(1, nn.BatchNorm1d(width, device=device))
        mods.insert(4, nn.BatchNorm1d(width, device=device))
    model = nn.Sequential(*mods)
    return model


class MySequential(nn.Sequential, MupModule):
    pass


def _generate_mup_MLP(width, bias=True, batchnorm=False, sampler="normal", device="cpu", **kwargs):
    mods = [
        MupLinear(1024, width, bias=bias, sampler=sampler, device=device, **kwargs),
        nn.ReLU(),
        MupLinear(width, width, bias=bias, sampler=sampler, device=device, **kwargs),
        nn.ReLU(),
        OutputLayer(width, 10, bias=bias, sampler=sampler, device=device, **kwargs),
    ]
    if batchnorm:
        mods.insert(1, nn.BatchNorm1d(width, device=device))
        mods.insert(4, nn.BatchNorm1d(width, device=device))
    model = MySequential(*mods)
    return model


class TestInfShape(unittest.TestCase):
    def test_infshape(self):
        infshape = InfShape([InfDim(None, 100), InfDim(128, 1024), InfDim(128, 128)])
        self.assertEqual(infshape.ninf(), 1)
        self.assertEqual(infshape.width_mult(), 8)

    def test_zip_infshape(self):
        infshape = zip_infshape([64, 128, 1024], [32, 128, 2048], fin_if_same=True)
        self.assertEqual(infshape.width_mult(), 2)
        infshape = zip_infshape([64, 128, 1024], [32, 128, 2048], fin_if_same=False)
        self.assertEqual(infshape.width_mult(), 2)


class TestShape(unittest.TestCase):
    def test_save_base_shape(self):
        model = _generate_mup_MLP(256, batchnorm=True)
        save_base_shapes(model, file=None)

    def test_make_base_shape(self):
        model = _generate_mup_MLP(256, bias=True, batchnorm=True)
        delta_model = _generate_mup_MLP(128, bias=True, batchnorm=True)
        make_base_shapes(model, delta_model)

    def _get_mlp_infshapes(self):
        base_model = _generate_mup_MLP(64, True, True)
        target_model = _generate_mup_MLP(128, True, True)
        set_base_shapes(target_model, base_model)
        return get_infshapes(target_model)

    def _get_mlp_infshapes_with_meta_device(self):
        base_model = _generate_mup_MLP(64, True, True, device="meta")
        target_model = _generate_mup_MLP(128, True, True, device="meta")
        set_base_shapes(target_model, base_model)
        return get_infshapes(target_model)

    def _get_mlp_infshapes_with_delta_model(self):
        base_model = _generate_mup_MLP(64, True, True)
        delta_model = _generate_mup_MLP(65, True, True)
        target_model = _generate_mup_MLP(128, True, True)
        set_base_shapes(target_model, base_model, delta=delta_model)
        return get_infshapes(target_model)

    def _get_mlp_infshapes_via_make(self):
        base_model = _generate_mup_MLP(64, True, True)
        delta_model = _generate_mup_MLP(65, True, True)
        base_infshapes = make_base_shapes(base_model, delta_model)
        target_model = _generate_mup_MLP(128, True, True)
        set_base_shapes(target_model, base_infshapes)
        return get_infshapes(target_model)

    def _get_mlp_infshapes_via_get_shapes(self):
        base_model = _generate_mup_MLP(64, True, True)
        target_model = _generate_mup_MLP(128, True, True)
        set_base_shapes(target_model, get_shapes(base_model))
        return get_infshapes(target_model)

    def _get_mlp_infshapes_bad(self):
        base_model = _generate_mup_MLP(64, True, True)
        target_model = _generate_mup_MLP(128, True, True)
        set_base_shapes(target_model, base_model, delta=base_model)
        return get_infshapes(target_model)

    def test_set_base_shape(self):
        self.assertEqual(self._get_mlp_infshapes(), self._get_mlp_infshapes_with_meta_device())
        self.assertEqual(self._get_mlp_infshapes(), self._get_mlp_infshapes_with_delta_model())
        self.assertEqual(self._get_mlp_infshapes(), self._get_mlp_infshapes_via_make())
        self.assertEqual(self._get_mlp_infshapes(), self._get_mlp_infshapes_via_get_shapes())
        self.assertNotEqual(self._get_mlp_infshapes(), self._get_mlp_infshapes_bad())


class TestInitialization(unittest.TestCase):
    def test_muplinear_const(self):
        layer = MupLinear(64, 256, bias=False, sampler="normal", std=0.02)
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertAlmostEqual(layer.weight.mean().item(), 0.0, delta=0.001)
        self.assertAlmostEqual(layer.weight.std().item(), 0.02, delta=0.001)
        layer.mup_initial(mode="mup")
        self.assertAlmostEqual(layer.weight.mean().item(), 0.0, delta=0.001)
        self.assertAlmostEqual(layer.weight.std().item(), 0.02 / math.sqrt(2), delta=0.001)

        layer = MupLinear(64, 256, bias=False, sampler="uniform", a=-1.0, b=1.0)
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertLessEqual(layer.weight.abs().max().item(), 1.0)
        layer.mup_initial(mode="mup")
        self.assertLessEqual(layer.weight.abs().max().item(), 1.0 / math.sqrt(2))

    def test_muplinear_xavier(self):
        layer = MupLinear(64, 256, bias=False, sampler="xavier_normal", gain=0.5)
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertAlmostEqual(layer.weight.mean().item(), 0.0, delta=0.001)
        self.assertAlmostEqual(layer.weight.std().item(), 0.5 * math.sqrt(2.0 / (64 + 256)), delta=0.001)
        layer.mup_initial(mode="mup")
        self.assertAlmostEqual(layer.weight.mean().item(), 0.0, delta=0.001)
        self.assertAlmostEqual(layer.weight.std().item(), 0.5 * math.sqrt(2.0 / (64 + 256)), delta=0.001)

        layer = MupLinear(64, 256, bias=False, sampler="xavier_uniform", gain=0.5)
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertLessEqual(layer.weight.abs().max().item(), 0.5 * math.sqrt(6.0 / (64 + 256)))
        layer.mup_initial(mode="mup")
        self.assertLessEqual(layer.weight.abs().max().item(), 0.5 * math.sqrt(6.0 / (64 + 256)))

    def test_muplinear_kaiming(self):
        # normal, infdim = 2
        layer = MupLinear(64, 256, bias=False, sampler="kaiming_normal", a=0)
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertAlmostEqual(layer.weight.mean().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(layer.weight.std().item(), math.sqrt(2.0 / 64), delta=0.01)
        layer.mup_initial(mode="mup")
        self.assertAlmostEqual(layer.weight.mean().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(layer.weight.std().item(), math.sqrt(2.0 / 64), delta=0.01)

        # normal, infdim = 1
        layer = MupLinear(64, 256, bias=False, sampler="kaiming_normal", a=0)
        layer.weight.infshape = InfShape([InfDim(256, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertAlmostEqual(layer.weight.mean().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(layer.weight.std().item(), math.sqrt(2.0 / 64), delta=0.01)
        layer.mup_initial(mode="mup")
        self.assertAlmostEqual(layer.weight.mean().item(), 0.0, delta=0.01)
        self.assertAlmostEqual(layer.weight.std().item(), math.sqrt(4.0 / 64), delta=0.01)

        # uniform
        layer = MupLinear(64, 256, bias=False, sampler="kaiming_uniform", a=0)
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertLessEqual(layer.weight.abs().max().item(), math.sqrt(3.0 * 2.0 / 64))
        layer.mup_initial(mode="mup")
        self.assertLessEqual(layer.weight.abs().max().item(), math.sqrt(3.0 * 2.0 / 64))

    def test_muplinear_with_bias(self):
        layer = MupLinear(64, 256, bias=True, bias_zero_init=True)
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertAlmostEqual(layer.bias.mean().item(), 0.0, delta=0.001)
        self.assertAlmostEqual(layer.bias.std().item(), 0.0, delta=0.001)
        layer.mup_initial(mode="mup")
        self.assertAlmostEqual(layer.bias.mean().item(), 0.0, delta=0.001)
        self.assertAlmostEqual(layer.bias.std().item(), 0.0, delta=0.001)

        layer = MupLinear(64, 256, bias=True, bias_zero_init=False)
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertLessEqual(layer.bias.max().item(), 1 / math.sqrt(64))
        layer.mup_initial(mode="mup")
        self.assertLessEqual(layer.bias.max().item(), math.sqrt(2) / math.sqrt(64))

    def test_muplinear_with_given_init_method(self):
        def init_method(std):
            def init_(tensor):
                return nn.init.normal_(tensor, mean=0.0, std=std)

            return init_

        layer = MupLinear(
            64, 256, bias=True, init_weight_method=init_method(0.001), init_bias_method=init_method(0.002)
        )
        layer.weight.infshape = InfShape([InfDim(128, 256), InfDim(32, 64)])
        layer.mup_initial(mode="sp")
        self.assertAlmostEqual(layer.weight.std().item(), 0.001, delta=0.001)
        self.assertAlmostEqual(layer.bias.std().item(), 0.002, delta=0.001)
        layer.mup_initial(mode="mup")
        self.assertAlmostEqual(layer.weight.std().item(), 0.001, delta=0.001)
        self.assertAlmostEqual(layer.bias.std().item(), 0.002, delta=0.001)

    def test_qkvlayer(self):
        dim = 64
        layer = QKVLayer(dim, dim * 3, q_zero_init=True)
        layer.weight.infshape = InfShape([InfDim(dim / 2 * 3, dim * 3), InfDim(dim / 2, dim)])
        layer.mup_initial(mode="sp")
        self.assertEqual(layer.weight.data[:dim, :].sum().item(), 0.0)
        self.assertNotEqual(layer.weight.data[dim : dim * 3, :].sum().item(), 0.0)
        layer.mup_initial(mode="mup")
        self.assertEqual(layer.weight.data[:dim, :].sum().item(), 0.0)
        self.assertNotEqual(layer.weight.data[dim : dim * 3, :].sum().item(), 0.0)

    def test_qlayer(self):
        dim = 64
        layer = QLayer(dim, dim * 3, q_zero_init=True)
        layer.weight.infshape = InfShape([InfDim(dim / 2 * 3, dim * 3), InfDim(dim / 2, dim)])
        layer.mup_initial(mode="sp")
        self.assertEqual(layer.weight.data.sum().item(), 0.0)
        layer.mup_initial(mode="mup")
        self.assertEqual(layer.weight.data.sum().item(), 0.0)

        layer = QLayer(dim, dim * 3, q_zero_init=False)
        layer.mup_initial(mode="sp")
        self.assertNotEqual(layer.weight.data.sum().item(), 0.0)

    def test_output_layer(self):
        dim = 64
        layer = OutputLayer(dim, 100, weight_zero_init=True)
        layer.weight.infshape = InfShape([InfDim(100, 100), InfDim(dim / 2, dim)])
        layer.mup_initial(mode="sp")
        self.assertEqual(layer.weight.sum().item(), 0.0)
        layer.mup_initial(mode="mup")
        self.assertEqual(layer.weight.sum().item(), 0.0)

        layer = OutputLayer(
            dim, 100, bias=True, weight_zero_init=False, bias_zero_init=False, sampler="normal", std=0.02
        )
        layer.weight.infshape = InfShape([InfDim(100, 100), InfDim(dim / 2, dim)])
        layer.set_width_mult()
        self.assertEqual(layer._width_mult, 2.0)
        layer.mup_initial(mode="sp")
        self.assertAlmostEqual(layer.weight.std().item(), 0.02, delta=0.001)
        self.assertLessEqual(layer.bias.abs().max().item(), 1.0 / math.sqrt(64))
        layer.mup_initial(mode="mup")
        self.assertAlmostEqual(layer.weight.std().item(), 0.02 * math.sqrt(2.0), delta=0.001)
        self.assertLessEqual(layer.bias.abs().max().item(), math.sqrt(2.0) / math.sqrt(64))

    def test_mlp_with_sp_init(self):
        # normal model
        torch.manual_seed(0)
        model = _generate_MLP(64, bias=True, batchnorm=False)
        _init_model(model, std=0.02)
        # mup model
        torch.manual_seed(0)
        mup_model = _generate_mup_MLP(64, bias=True, batchnorm=False, sampler="normal", std=0.02, bias_zero_init=False)
        set_base_shapes(mup_model, None)
        mup_model.mup_initial(mode="sp")

        diff = 0.0
        for (_, mp), (_, p) in zip(mup_model.named_parameters(), model.named_parameters()):
            diff += (mp - p).abs().sum().item()
        self.assertEqual(diff, 0.0)

    def test_mlp_with_mup_init_at_base_width(self):
        # normal model
        torch.manual_seed(0)
        model = _generate_MLP(64, bias=True, batchnorm=False)
        _init_model(model, std=0.02)
        # mup model
        torch.manual_seed(0)
        mup_model = _generate_mup_MLP(64, bias=True, batchnorm=False, sampler="normal", std=0.02, bias_zero_init=False)
        set_base_shapes(mup_model, None)
        mup_model.mup_initial(mode="mup")

        diff = 0.0
        for (_, mp), (_, p) in zip(mup_model.named_parameters(), model.named_parameters()):
            diff += (mp - p).abs().sum().item()
        self.assertEqual(diff, 0.0)

    def test_mlp_with_mup_init_at_diff_width(self):
        # normal model
        torch.manual_seed(0)
        model = _generate_MLP(64, bias=True, batchnorm=False)
        _init_model(model, std=0.02)
        # mup model
        torch.manual_seed(0)
        mup_model = _generate_mup_MLP(64, bias=True, batchnorm=False, sampler="normal", std=0.02, bias_zero_init=False)
        base_model = _generate_mup_MLP(32, bias=True, batchnorm=False, sampler="normal", std=0.02, bias_zero_init=False)
        set_base_shapes(mup_model, base_model)
        mup_model.mup_initial(mode="mup")

        same_name = ["0.weight", "0.bias"]
        scale_down_name = ["2.weight"]
        scale_up_name = ["2.bias", "4.weight", "4.bias"]
        for (name, mp), (_, p) in zip(mup_model.named_parameters(), model.named_parameters()):
            with self.subTest(name=name):
                if name in same_name:
                    self.assertAlmostEqual(mp.std().item(), p.std().item(), delta=0.001)
                elif name in scale_down_name:
                    self.assertAlmostEqual(mp.std().item(), p.std().item() / math.sqrt(2), delta=0.001)
                elif name in scale_up_name:
                    self.assertAlmostEqual(mp.std().item(), p.std().item() * math.sqrt(2), delta=0.001)


class TestMuOptim(unittest.TestCase):
    def _train(self, input, target, model, optimizer):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()

        def closure():
            output = model(input)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            return loss

        initial_value = closure().item()
        for _ in range(10):
            loss = optimizer.step(closure)
            optimizer.zero_grad()

        self.assertLess(loss.item(), initial_value)

    @unittest.skipIf(not torch.cuda.is_available(), "no gpu found here.")
    def test_MuSGD(self):
        # 定义模型
        model = _generate_mup_MLP(64, bias=True, batchnorm=True, sampler="normal", std=0.02, bias_zero_init=False)
        base_model = _generate_mup_MLP(32, bias=True, batchnorm=True, sampler="normal", std=0.02, bias_zero_init=False)
        set_base_shapes(model, base_model)
        model.mup_initial(mode="mup")
        # 定义优化器
        opt = MuSGD(model.parameters(), lr=0.1, weight_decay=0.0)

        self._train(
            input=torch.randn([2, 1024]),
            target=torch.tensor([0, 1]),
            model=model,
            optimizer=opt,
        )

    @unittest.skipIf(not torch.cuda.is_available(), "no gpu found here.")
    def test_MuAdam(self):
        # 定义模型
        model = _generate_mup_MLP(64, bias=True, batchnorm=True, sampler="normal", std=0.02, bias_zero_init=False)
        base_model = _generate_mup_MLP(32, bias=True, batchnorm=True, sampler="normal", std=0.02, bias_zero_init=False)
        set_base_shapes(model, base_model)
        model.mup_initial(mode="mup")
        # 定义优化器
        opt = MuAdam(model.parameters(), lr=0.1, weight_decay=0.0)

        self._train(
            input=torch.randn([2, 1024]),
            target=torch.tensor([0, 1]),
            model=model,
            optimizer=opt,
        )


if __name__ == "__main__":
    unittest.main()
