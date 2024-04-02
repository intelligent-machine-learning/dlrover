import copy
import math
import os
import tempfile
import unittest
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import torch.nn as nn

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.common.util_func import find_free_port
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload, reload_meta_module
from atorch.utils.version import torch_version


def init_dist(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    if torch.cuda.is_available():
        atorch.init_distributed("nccl")
    else:
        atorch.init_distributed("gloo")


def init_bert_params(module):
    def normal_(data):
        data.copy_(data.normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, MultiwayNetwork):
        normal_(module.A.weight.data)
        normal_(module.B.weight.data)


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1
        self.p = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class FFN(nn.Module):
    def __init__(self, dim, nlayers):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([])
        for i in range(nlayers):
            self.layers.append(MultiwayNetwork(nn.Linear(dim, dim)))
        self.layers.apply(init_bert_params)
        self.ln = MultiwayNetwork(nn.LayerNorm(dim))

    def forward(self, x):
        x = self.fc1(x)
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)


class HoldModule(nn.Module):
    def __init__(
        self,
        modules,
    ):
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.mods:
            x = module(x)
        return x


class MyModel(nn.Module):
    def __init__(self, nlayers, dim):
        super().__init__()
        self.ffn = FFN(dim, nlayers)
        self.ffn2 = FFN(dim, nlayers)
        self.ln = nn.LayerNorm(dim)
        self.layers = HoldModule([self.ffn, self.ffn2])
        init_scale = math.sqrt(math.log(nlayers * 2))
        self.p = nn.Parameter(torch.tensor(1.0))
        for name, p in self.named_parameters():
            if "bias" in name:
                p.data.zero_()
            elif "weight" in name:
                p.data.mul_(init_scale)

    def forward(self, x):
        return self.ln(self.layers(x))


class TestMetaInit(unittest.TestCase):
    embed_dim = 10

    def setUp(self) -> None:
        return super().setUp()

    def test_init_MWN_params(self):
        with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
            model = MyModel(5, self.embed_dim)
            for name, params in model.named_parameters():
                self.assertTrue(params.device.type == "cpu")
            for _, p in model.layers.mods[0].named_parameters():
                p.requires_grad_(False)
        for _, p in model.layers.mods[0].named_parameters():
            self.assertFalse(p.requires_grad)
        device2name = defaultdict(list)
        self.assertEqual(3, len(list(model.ffn.layers[0].modules())))  # self,A,B
        for name, params in model.named_parameters():
            device2name[params.device.type].append(name)
        for name, params in model.named_parameters():
            self.assertTrue(params.device.type == "meta")

        self.assertEqual(1, len(model._parameters))  # p
        for name in model._parameters:
            self.assertTrue(name in device2name["meta"])
        mod1 = nn.Linear(10, 10)
        mod2 = copy.deepcopy(mod1)

        reload_meta_module(model, "cpu")
        for _, p in model.layers.mods[0].named_parameters():
            self.assertFalse(p.requires_grad)
        self.assertFalse(mod2 in [mod1])

    def test_init_MWN_params2(self):
        with tempfile.TemporaryFile() as fp:
            with init_empty_weights_with_disk_offload(ignore_tie_weights=True):
                model = MyModel(5, self.embed_dim)
                torch.save(model.state_dict(), fp)
            fp.seek(0)
            sd = torch.load(fp, map_location="cpu")
            model.load_state_dict(sd)
        device2name = defaultdict(list)
        for name, params in model.named_parameters():
            device2name[params.device].append(name)
        for name, params in model.named_parameters():
            self.assertTrue(params.device.type == "meta")
        # reload_meta_module(model, "cpu")


def _test_init_auto_accelerate(rank, world_size):
    init_dist(rank, world_size)
    embed_dim = 10
    with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
        model = MyModel(5, embed_dim)
    # simulate requires_grad=False
    for _, p in model.layers.mods[0].named_parameters():
        p.requires_grad_(False)
    reload_meta_module(model, "cpu")
    # TODO: fix when using "atorch_wrap_cls": (MultiwayNetwork,)  the test fails
    auto_accelerate(
        model,
        load_strategy=[
            (
                "fsdp",
                {"atorch_wrap_cls": {"nn.Linear"}, "use_orig_params": True},
            )
        ],
    )

    atorch.reset_distributed()


class TestInitAutoAccelerate(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() < 2 or torch_version() < (2, 0, 0),  # type: ignore
        "run with gpu_num >=2 torch.version > 2.0",
    )
    def test_init_auto_accelerate(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"  #
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            _test_init_auto_accelerate,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
