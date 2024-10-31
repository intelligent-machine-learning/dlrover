# flake8: noqa: E402
import os
import shutil
import unittest

import pytest
import torch

from atorch.utils.version import torch_version

torch = pytest.importorskip("torch", minversion="2.0.9")
if torch.version.git_version != "7bcf7da3a268b435777fe87c7794c382f444e86d" or not torch.cuda.is_available():
    pytest.skip("requires pytorch 2.1 stable release", allow_module_level=True)

is_torch_bigger_than_24 = False
if torch_version() >= (2, 4, 0):  # type: ignore
    is_torch_bigger_than_24 = True
else:
    is_torch_bigger_than_24 = False

import torch.multiprocessing as mp
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.wrap import CustomPolicy

import atorch
from atorch.auto import auto_accelerate
from atorch.common.util_func import divide, find_free_port
from atorch.distributed.distributed import create_parallel_group, parallel_group, parallel_group_and_ranks

# from atorch.utils.fsdp_async_ckpt_util import save_checkpoint
from atorch.utils.fsdp_init_util import clear_fsdp_patch_init, patch_fsdp_init
from atorch.utils.fsdp_save_util import ShardOptim, save_fsdp_flat_param, save_fsdp_optim_param

torch.distributed.fsdp._runtime_utils._validate_and_get_hybrid_shard_state = lambda x: None


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
        y = self.bw2(self.bw1(x))
        return self.experts(y)


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


cls_to_fsdp_group_config = {}


def init_moe_group(ep_size=2, ddp1=None, ddp2=None, use_device_mesh=False):
    global cls_to_fsdp_group_config

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    world_size = torch.distributed.get_world_size()
    # ep_size = divide(world_size, 2)
    if ddp1 is not None and ddp1 > 1:
        fsdp_mode = ([("data", divide(world_size, ddp1)), ("ddp1", ddp1)], None)
    else:
        fsdp_mode = ([("data", torch.distributed.get_world_size())], None)

    create_parallel_group(fsdp_mode, use_device_mesh=use_device_mesh)
    if ddp1 is not None and ddp1 > 1:
        cls_to_fsdp_group_config[DummyBlock] = (parallel_group("data"), parallel_group("ddp1"))
    else:
        cls_to_fsdp_group_config[DummyBlock] = parallel_group("data")

    if ep_size > 1:
        if ddp2 is not None and ddp2 > 1:
            ep_mode = ([("expert", ep_size), ("expert_fsdp", divide(world_size, ep_size * ddp2)), ("ddp2", ddp2)], None)
        else:
            ep_mode = ([("expert", ep_size), ("expert_fsdp", divide(world_size, ep_size))], None)

        create_parallel_group(ep_mode, use_device_mesh=use_device_mesh)
        if ddp2 is not None and ddp2 > 1:
            cls_to_fsdp_group_config[DummyExperts] = (parallel_group("expert_fsdp"), parallel_group("ddp2"))
        else:
            cls_to_fsdp_group_config[DummyExperts] = parallel_group("expert_fsdp")
    else:
        expert_fsdp_mode = ([("expert_fsdp", world_size)], None)
        create_parallel_group(expert_fsdp_mode, use_device_mesh=use_device_mesh)
        cls_to_fsdp_group_config[DummyExperts] = parallel_group("expert_fsdp")

    return cls_to_fsdp_group_config


def moe_hsdp_policy_fn(module):
    global cls_to_fsdp_group_config
    cls_to_fsdp_group = {DummyBlock: parallel_group("data"), DummyExperts: parallel_group("expert_fsdp")}
    # cls_to_fsdp_group = {DummyBlock: parallel_group("data")}

    if module.__class__ in cls_to_fsdp_group:
        pg = cls_to_fsdp_group_config[module.__class__]
        if isinstance(pg, tuple):
            return {"process_group": pg, "sharding_strategy": ShardingStrategy.HYBRID_SHARD}
        return {"process_group": pg}
    return False


def _init_group_and_model(rank, world_size, hidden_size, ddp1_size=None, ddp2_size=None, ep_size=2):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group(ep_size=ep_size, ddp1=ddp1_size, ddp2=ddp2_size)
    m = get_model(hidden_size)
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
        "auto_wrap_policy": CustomPolicy(moe_hsdp_policy_fn),
    }
    if ddp1_size is not None and ddp1_size > 1:
        fsdp_config.update(
            {
                "process_group": (parallel_group("data"), parallel_group("ddp1")),
                "sharding_strategy": ShardingStrategy.HYBRID_SHARD,
            }
        )

    strategy = [("fsdp", fsdp_config)]

    status, result, best_strategy = auto_accelerate(
        m,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model = result.model
    optim = result.optim

    loss = model(get_input(hidden_size)).mean()
    loss.backward()
    optim.step()

    return model, optim, fsdp_config


def run_module_gt(rank, world_size, hidden_size, path, ddp1_size=None, ddp2_size=None, async_save=False, ep_size=2):
    model, optim, _ = _init_group_and_model(
        rank=rank,
        world_size=world_size,
        hidden_size=hidden_size,
        ddp1_size=ddp1_size,
        ddp2_size=ddp2_size,
        ep_size=ep_size,
    )

    _, ranks = parallel_group_and_ranks("ddp1")
    if ranks is None or rank == ranks[0]:
        if async_save:
            pass
            # pg = parallel_group("data")
            # save_checkpoint(step=1, model=model, optimizer=optim, path=path, group=pg)
        else:
            save_fsdp_flat_param(model, path)
            save_fsdp_optim_param(model, optim, path)


def run_module_load(rank, world_size, hidden_size, gt_ckpt, ddp1_size=None, ddp2_size=None, check_strict=True):
    # ref_model
    ref_model, ref_optim, fsdp_config = _init_group_and_model(
        rank=rank, world_size=world_size, hidden_size=hidden_size, ddp1_size=ddp1_size, ddp2_size=ddp2_size
    )
    print("ref_model:", ref_model.sharding_strategy)

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
    strategy = [("fsdp", fsdp_config)]

    status, result, best_strategy = auto_accelerate(
        load_m,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    load_model = result.model
    print("load_model:", load_model.sharding_strategy)
    load_optim = result.optim
    sm = ShardOptim(gt_ckpt)
    reshard_optim_state = sm.reshard_optim_state_dict(load_model)
    load_optim.load_state_dict(reshard_optim_state)
    assert_same_sd(ref_model.state_dict(), load_model.state_dict(), strict=check_strict)
    assert_same_sd(ref_optim.state_dict(), load_optim.state_dict(), strict=check_strict)


def run_module_double_load(rank, world_size, hidden_size, gt_ckpt):
    # ref_model
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group()
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
        "auto_wrap_policy": CustomPolicy(moe_hsdp_policy_fn),
    }
    # patch fsdp init load need sync_module_states to be False and to_empty param_init_fn
    fsdp_config["sync_module_states"] = False
    fsdp_config["param_init_fn"] = lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
    strategy = [("fsdp", fsdp_config)]

    # load_model
    load_model1 = get_model(hidden_size, meta=True)
    patch_fsdp_init(
        gt_ckpt,
        (DummyBlock, DummyExperts),
        load_model1,
    )

    _, result1, _ = auto_accelerate(
        load_model1,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model1 = result1.model
    optim1 = result1.optim
    print("load_model1:", model1.sharding_strategy)

    # clear patch
    clear_fsdp_patch_init()

    load_model2 = get_model(hidden_size, meta=True)
    patch_fsdp_init(
        gt_ckpt,
        (DummyBlock, DummyExperts),
        load_model2,
    )
    _, result2, _ = auto_accelerate(
        load_model2,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model2 = result2.model
    optim2 = result2.optim
    print("load_model2:", model2.sharding_strategy)

    # train
    input_data = get_input(hidden_size)
    loss1 = model1(input_data).mean()
    loss1.backward()
    optim1.step()

    loss2 = model2(input_data).mean()
    loss2.backward()
    optim2.step()


def run_module_fsdp2(rank, world_size, hidden_size, gt_ckpt):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    init_moe_group(use_device_mesh=True, ep_size=1)
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "auto_wrap_policy": CustomPolicy(moe_hsdp_policy_fn),
    }
    # patch fsdp init load need sync_module_states to be False and to_empty param_init_fn
    fsdp_config["sync_module_states"] = False
    fsdp_config["param_init_fn"] = lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)

    amp_config = {"dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16}
    strategy = [("fsdp", fsdp_config), ("amp_native", amp_config)]

    # load_model
    load_model1 = get_model(hidden_size, meta=True)
    patch_fsdp_init(
        gt_ckpt,
        (DummyBlock, DummyExperts),
        load_model1,
    )

    _, result1, _ = auto_accelerate(
        load_model1,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        load_strategy=strategy,
        optim_param_func=optim_param_func,
        ignore_dryrun_on_load_strategy=True,
    )
    model1 = result1.model
    optim1 = result1.optim

    # train
    input_data = get_input(hidden_size)
    loss1 = model1(input_data).mean()
    loss1.backward()
    optim1.step()


def assert_same_sd(ref_sd, load_sd, strict=True):
    assert set(ref_sd.keys()) == set(load_sd.keys()), (
        f"{[k for k in ref_sd.keys() if k not in load_sd.keys()]} "
        f"{[k for k in load_sd.keys() if k not in ref_sd.keys()]}"
    )
    for k in ref_sd.keys():
        if isinstance(ref_sd[k], dict):
            assert_same_sd(ref_sd[k], load_sd[k], strict)
        elif isinstance(ref_sd[k], torch.Tensor):
            if strict:
                assert torch.all(ref_sd[k] == load_sd[k]), f"{k}\nref_sd\n{ref_sd[k]}\nload_sd\n{load_sd[k]}"
            else:
                assert torch.allclose(
                    ref_sd[k], load_sd[k], rtol=1e-02
                ), f"{k}\nref_sd\n{ref_sd[k]}\nload_sd\n{load_sd[k]}"


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 8,
    "Must have at least 8 GPUs for gpu test",
)
class HSDPMoEShardSaveLoadTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(HSDPMoEShardSaveLoadTest, self).__init__(methodName)
        self.world_size = world_size or 8
        self.hidden_size = hidden_size or 64
        self.gt_ckpt = f"/tmp/hsdp_moe_save_load_test/gt"

    def setUp(self):
        atorch.reset_distributed()

    def _prepare_toy_save(self, ddp1_size=None, ddp2_size=None, async_save=False):
        """This test will save toy module"""

        if os.path.exists("/tmp/hsdp_moe_save_load_test/"):
            shutil.rmtree("/tmp/hsdp_moe_save_load_test/")
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_module_gt,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt, ddp1_size, ddp2_size, async_save),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_save_fsdp(self):
        self._prepare_toy_save()

    def test_save_hsdp(self):
        self._prepare_toy_save(ddp1_size=2, ddp2_size=2)

    # def test_async_save_fsdp(self):
    #     self._prepare_toy_save(async_save=True)

    # def test_async_save_hsdp(self):
    #     self._prepare_toy_save(ddp1_size=2, ddp2_size=2, async_save=True)

    def test_hsdp_load_from_hsdp_ckpt(self, async_save=False):
        """This test will load toy module"""
        ddp1_size = 2
        ddp2_size = 2
        self._prepare_toy_save(ddp1_size, ddp2_size, async_save=async_save)

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt, ddp1_size, ddp2_size),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    # def test_hsdp_load_from_hsdp_async_save_ckpt(self):
    #     self.test_hsdp_load_from_hsdp_ckpt(async_save=True)

    def test_hsdp_load_from_fsdp_ckpt(self, async_save=False):
        """This test will load toy module"""
        self._prepare_toy_save(async_save=async_save)

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        ddp1_size = 2
        ddp2_size = 2
        mp.spawn(
            run_module_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt, ddp1_size, ddp2_size, False),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    # def test_hsdp_load_from_fsdp_async_save_ckpt(self):
    #     self.test_hsdp_load_from_fsdp_ckpt(async_save=True)


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    "Must have at least 4 GPUs for gpu test",
)
class FSDPMoEDPOTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDPMoEDPOTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64
        self.gt_ckpt = f"/tmp/fsdp_moe_dpo_test/gt"

    def setUp(self):
        atorch.reset_distributed()

    def _prepare_toy_save(self):
        """This test will save toy module"""

        if os.path.exists("/tmp/fsdp_moe_dpo_test/"):
            shutil.rmtree("/tmp/fsdp_moe_dpo_test/")
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_module_gt,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp_doule_init_and_accerate(self):
        self._prepare_toy_save()

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_double_load,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4 or not is_torch_bigger_than_24,
    "Must have at least 4 GPUs for gpu test",
)
class FSDPSFTTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDPSFTTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64
        self.gt_ckpt = f"/tmp/fsdp_moe_sft_test/gt"

    def setUp(self):
        atorch.reset_distributed()

    def _prepare_toy_save(self):
        """This test will save toy module"""

        if os.path.exists("/tmp/fsdp_moe_sft_test/"):
            shutil.rmtree("/tmp/fsdp_moe_sft_test/")
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_module_gt,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt, None, None, False, 1),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()

    def test_fsdp2(self):
        self._prepare_toy_save()

        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)

        mp.spawn(
            run_module_fsdp2,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, self.gt_ckpt),
            join=True,
            daemon=False,
            start_method="spawn",
        )
        atorch.reset_distributed()
