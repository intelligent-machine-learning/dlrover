# flake8: noqa: E402
import os
import shutil
import unittest

import pytest
import torch

torch = pytest.importorskip("torch", minversion="2.0.9")
if torch.version.git_version != "7bcf7da3a268b435777fe87c7794c382f444e86d" or not torch.cuda.is_available():
    pytest.skip("requires pytorch 2.1 stable release", allow_module_level=True)

import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import CustomPolicy

import atorch
from atorch.common.util_func import divide, find_free_port
from atorch.distributed.distributed import create_parallel_group, parallel_group
from atorch.utils.fsdp_init_util import patch_fsdp_init
from atorch.utils.fsdp_save_util import ShardOptim, save_fsdp_flat_param, save_fsdp_optim_param


# Toy for moe
class DummyExperts(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.w2 = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return self.w2(self.w1(x))


class DummyBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.bw1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.bw2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.experts = DummyExperts(hidden_size)

    def forward(self, x):
        return self.experts(self.bw2(self.bw1(x)))


class DummyModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mw1 = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.mw2 = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.layers = torch.nn.ModuleList(DummyBlock(hidden_size) for _ in range(3))

    def forward(self, x):
        x = self.mw2(self.mw1(x))
        for layer in self.layers:
            x = layer(x)
        return x


def get_model(hidden_size, seed=123, meta=False):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    if meta:
        with torch.device("meta"):
            model = DummyModel(hidden_size)
    else:
        model = DummyModel(hidden_size).cuda()
    return model


def get_input(hidden_size, seed=123):
    diff_seed = seed + torch.distributed.get_rank()
    torch.cuda.manual_seed(diff_seed)
    torch.manual_seed(diff_seed)
    return torch.randn(4, hidden_size, device=torch.device("cuda"))


def get_optim(model):
    def optim_param_func(model):
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    optim = torch.optim.AdamW(optim_param_func(model), lr=0.001)
    return optim


def init_moe_group():
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    world_size = torch.distributed.get_world_size()
    ep_size = divide(world_size, 2)
    fsdp_mode = ([("data", torch.distributed.get_world_size())], None)
    create_parallel_group(fsdp_mode)
    ep_mode = ([("expert", ep_size), ("expert_fsdp", 2)], None)
    create_parallel_group(ep_mode)


def moe_fsdp_policy_fn(module):
    if isinstance(module, DummyBlock):
        # non experts fsdp wrap
        return {"process_group": parallel_group("data")}
        # return True
    elif isinstance(module, DummyExperts):
        # experts fsdp wrap
        return {"process_group": parallel_group("expert_fsdp")}
    return False


def run_module_gt(rank, world_size, hidden_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group()
    m = get_model(hidden_size)
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
        "auto_wrap_policy": CustomPolicy(moe_fsdp_policy_fn),
    }
    model = FSDP(m, **fsdp_config)
    optim = get_optim(model)
    loss = model(get_input(hidden_size)).mean()
    loss.backward()
    optim.step()
    path = f"/tmp/fsdp_moe_save_load_test/{world_size}/gt"
    save_fsdp_flat_param(model, path)
    save_fsdp_optim_param(model, optim, path)


def run_module_load(rank, world_size, hidden_size, gt_ckpt):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group()

    # ref_model
    ref_m = get_model(hidden_size)
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
        "auto_wrap_policy": CustomPolicy(moe_fsdp_policy_fn),
    }
    ref_model = FSDP(ref_m, **fsdp_config)
    ref_optim = get_optim(ref_model)
    loss = ref_model(get_input(hidden_size)).mean()
    loss.backward()
    ref_optim.step()

    # load_model
    load_m = get_model(hidden_size, meta=True)
    patch_fsdp_init(
        gt_ckpt,
        (DummyBlock, DummyExperts),
        load_m,
    )
    # patch fsdp init load need sync_module_states to be False and to_empty param_init_fn
    fsdp_config["sync_module_states"] = False
    fsdp_config["param_init_fn"] = lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
    load_model = FSDP(load_m, **fsdp_config)  # ckpt load already done here
    load_optim = get_optim(load_model)
    sm = ShardOptim(gt_ckpt)
    reshard_optim_state = sm.reshard_optim_state_dict(load_model)
    load_optim.load_state_dict(reshard_optim_state)
    assert_same_sd(ref_model.state_dict(), load_model.state_dict())
    assert_same_sd(ref_optim.state_dict(), load_optim.state_dict())


def assert_same_sd(ref_sd, load_sd):
    assert set(ref_sd.keys()) == set(load_sd.keys()), (
        f"{[k for k in ref_sd.keys() if k not in load_sd.keys()]} "
        f"{[k for k in load_sd.keys() if k not in ref_sd.keys()]}"
    )
    for k in ref_sd.keys():
        if isinstance(ref_sd[k], dict):
            assert_same_sd(ref_sd[k], load_sd[k])
        elif isinstance(ref_sd[k], torch.Tensor):
            assert torch.all(ref_sd[k] == load_sd[k]), f"{k}\nref_sd\n{ref_sd[k]}\nload_sd\n{load_sd[k]}"


class FSDPMoEShardSaveLoadTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDPMoEShardSaveLoadTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64
        self.test_dir_prefix = str(self.world_size)
        self.gt_ckpt = f"/tmp/fsdp_moe_save_load_test/{self.test_dir_prefix}/gt"
        if os.path.exists("/tmp/fsdp_moe_save_load_test/{self.test_dir_prefix}"):
            shutil.rmtree("/tmp/fsdp_moe_save_load_test/{self.test_dir_prefix}")
        self._prepare_toy_save()

    def setUp(self):
        atorch.reset_distributed()

    def _prepare_toy_save(self):
        """This test will save toy module"""
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_module_gt,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Must have at least 4 GPUs for gpu test",
    )
    def test_toy_load(self):
        """This test will load toy module"""
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)
        mp.spawn(
            run_module_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()
