# flake8: noqa: E402
import functools
import operator
import os
import shutil
import time
import unittest

import pytest
import torch

torch = pytest.importorskip("torch", minversion="2.0.9")
if torch.version.git_version != "7bcf7da3a268b435777fe87c7794c382f444e86d" or not torch.cuda.is_available():
    pytest.skip("requires pytorch 2.1 stable release", allow_module_level=True)
from unittest import mock

import torch.multiprocessing as mp
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.flat_param import FlatParamHandle, HandleShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import atorch
from atorch.common.util_func import find_free_port
from atorch.utils.fsdp_async_ckpt_util import save_checkpoint as async_save_checkpoint
from atorch.utils.fsdp_save_util import ShardOptim, ShardTensorUtil, save_fsdp_flat_param, save_fsdp_optim_param

VOCAB_SIZE = 373


class MLP(nn.Module):
    index = 0

    def __init__(self, hidden):
        """
        Args:
            in_feature (int): size of input feature.
            out_feature (int): size of output feature.
        """
        super().__init__()
        self.up = torch.nn.Linear(hidden, hidden * 4)
        self.down = torch.nn.Linear(hidden * 4, hidden)
        self.register_buffer("fake_buffer", torch.ones(hidden) * MLP.index)
        MLP.index += 1

    def forward(self, x):
        return self.down(self.up(x)) + self.fake_buffer


class ToyModel(nn.Module):
    def __init__(self, hidden_size):
        """
        Args:
            in_feature (int): size of input feature.
            out_feature (int): size of output feature.
        """
        super(ToyModel, self).__init__()
        vocab_size = VOCAB_SIZE
        hidden = hidden_size

        self.emb = nn.Embedding(vocab_size, hidden)
        self.linears = torch.nn.ModuleList([MLP(hidden) for _ in range(2)])
        self.lm_head = torch.nn.Linear(hidden, vocab_size)
        self.emb.weight = self.lm_head.weight

    def forward(self, inputs):
        data = self.emb(inputs)
        for l in self.linears:
            data = l(data)
        return self.lm_head(data)


def run_module_gt(rank, world_size, hidden_size, asynced=False):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    m = ToyModel(hidden_size).cuda()
    ids = torch.randint(0, VOCAB_SIZE, (3, 2)).cuda()
    fp16_dtype = torch.float16
    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": True,
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
    }
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={MLP})
    model = FSDP(m, auto_wrap_policy=wrap_policy, **fsdp_config)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss = model(ids).mean()
    loss.backward()
    optim.step()
    if asynced:
        path = f"/tmp/fsdp_speedup_init_test/{world_size}/async_gt"
        async_save_checkpoint(1, model, optim, path)
        time.sleep(5)  # Wait async saving.
    else:
        path = f"/tmp/fsdp_speedup_init_test/{world_size}/gt"
        save_fsdp_flat_param(model, path)
        save_fsdp_optim_param(model, optim, path)


def run_module_reshard(rank, world_size, ckpt_path, hidden_size, test_dir_prefix, asynced=False):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    with torch.device("meta"):
        m = ToyModel(hidden_size)
    from atorch.utils.fsdp_init_util import patch_fsdp_init

    patch_fsdp_init(ckpt_path, MLP, m)
    fp16_dtype = torch.float16

    fsdp_config = {
        "use_orig_params": True,
        "sync_module_states": False,
        "param_init_fn": lambda module: module.to_empty(device=torch.device("cuda"), recurse=False),
        "mixed_precision": MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype),
    }
    wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={MLP})
    model = FSDP(m, auto_wrap_policy=wrap_policy, **fsdp_config)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    sm = ShardOptim(ckpt_path)
    reshard_optim_state = sm.reshard_optim_state_dict(model)
    optim.load_state_dict(reshard_optim_state)
    if asynced:
        path = f"/tmp/fsdp_speedup_init_test/{test_dir_prefix}/reshard/{world_size}/async"
        async_save_checkpoint(1, model, optim, path)
        time.sleep(5)  # Wait async saving.
    else:
        path = f"/tmp/fsdp_speedup_init_test/{test_dir_prefix}/reshard/{world_size}"
        save_fsdp_flat_param(model, path)
        save_fsdp_optim_param(model, optim, path)


class FSDPShardSaveLoadTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDPShardSaveLoadTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64
        self.test_dir_prefix = str(self.world_size)
        self.gt_ckpt = f"/tmp/fsdp_speedup_init_test/{self.test_dir_prefix}/gt"
        if os.path.exists("/tmp/fsdp_speedup_init_test/{self.test_dir_prefix}"):
            shutil.rmtree("/tmp/fsdp_speedup_init_test/{self.test_dir_prefix}")
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

        # generate flat param ckpt
        def boot_inner(world_size):
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["NPROC_PER_NODE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(find_free_port())
            mp.spawn(
                run_module_reshard,
                nprocs=world_size,
                args=(world_size, self.gt_ckpt, self.hidden_size, self.test_dir_prefix),
                join=True,
                daemon=False,
                start_method="spawn",
            )
            atorch.reset_distributed()

        def load(ckpt_path):
            util = ShardTensorUtil(ckpt_path, 0, 1, device="cpu")
            buffers = {i: util.load_tensor_by_name(i) for i in util.buffers.keys()}
            params = {i: util.load_tensor_by_name(i) for i in util.param_meta.keys()}
            return {**buffers, **params}

        # load flat param ckpt which generated in step 1
        # save and reshard
        shard_numbers = [2, 3, 4]
        loaded_state_dict = []
        for shard in shard_numbers:
            boot_inner(shard)
            loaded_state_dict.append(load(f"/tmp/fsdp_speedup_init_test/{self.test_dir_prefix}/reshard/{shard}"))
        gt_state_dict = load(self.gt_ckpt)
        for state_dict in loaded_state_dict:
            for name, gt_value in gt_state_dict.items():
                # only save/load, all values should be equal.
                self.assertTrue((gt_value == state_dict[name]).all())

    def test_reshard(self):
        def test_inner(rank, world_size, hidden_size):
            pg = mock.Mock()
            pg.rank.return_value = rank
            pg.size.return_value = world_size
            ckpt_util = ShardTensorUtil(self.gt_ckpt, rank, world_size, device="cpu")
            module = ToyModel(hidden_size)
            ckpt_util.get_fsdp_init_order(module, {MLP}, build_fsdp_load_map=False)
            buffers = {i: ckpt_util.load_tensor_by_name(i) for i in ckpt_util.buffers.keys()}
            params = {i: ckpt_util.load_tensor_by_name(i) for i in ckpt_util.param_meta.keys()}
            weights = {**buffers, **params}
            weights["lm_head.weight"] = weights["emb.weight"]
            module.load_state_dict(weights)
            # flat param in ckpt is
            # OrderedDict([('linears.0', ['up.weight', 'up.bias', 'down.weight', 'down.bias']), ('linears.1', ['up.weight', 'up.bias', 'down.weight', 'down.bias']), ('', ['emb.weight', 'lm_head.bias'])])
            for fsdp_unit_name, inner_name in ckpt_util.init_module_names.items():
                if fsdp_unit_name:
                    fsdp = operator.attrgetter(fsdp_unit_name)(module)
                    params_to_flatten = list(fsdp.parameters())
                else:
                    # this is default fsdp unit
                    params_to_flatten = [module.emb.weight, module.lm_head.bias]
                default_config = {
                    "device": torch.device("cuda"),
                    "sharding_strategy": HandleShardingStrategy.FULL_SHARD,
                    "offload_params": False,
                    "mp_param_dtype": torch.float16,
                    "mp_reduce_dtype": torch.float16,
                    "keep_low_precision_grads": False,
                    "process_group": pg,
                    "use_orig_params": True,
                }
                handle = FlatParamHandle(params_to_flatten, module, **default_config)

                loaded_flat_param, pad = ckpt_util.load_flat_param_by_name(fsdp_unit_name, handle)
                handle.shard()
                handle_flat_param = handle.flat_param
                self.assertTrue((handle_flat_param == loaded_flat_param).all())

        for world_size in [2, 3, 4, 7, 23, 1344]:
            if world_size == 1344:
                sub_ranks = (0, 3, 5, 34)
            else:
                sub_ranks = range(world_size)
            for rank in sub_ranks:
                test_inner(rank, world_size, self.hidden_size)


class FSDPShardAsyncSaveLoadTest(unittest.TestCase):
    def __init__(self, methodName="runTest", world_size=None, hidden_size=None):
        super(FSDPShardAsyncSaveLoadTest, self).__init__(methodName)
        self.world_size = world_size or 4
        self.hidden_size = hidden_size or 64
        self.test_dir_prefix = str(self.world_size)
        self.gt_ckpt = f"/tmp/fsdp_speedup_init_test/{self.test_dir_prefix}/gt"
        if os.path.exists("/tmp/fsdp_speedup_init_test/{self.test_dir_prefix}"):
            shutil.rmtree("/tmp/fsdp_speedup_init_test/{self.test_dir_prefix}")
        self._prepare_toy_save()

    def setUp(self):
        atorch.reset_distributed()

    def _prepare_toy_save(self):
        """This test will save toy module"""
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["NPROC_PER_NODE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["NVTE_TORCH_COMPILE"] = str(0)
        mp.spawn(
            run_module_gt,
            nprocs=self.world_size,
            args=(self.world_size, self.hidden_size, True),
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

        # generate flat param ckpt
        def boot_inner(world_size):
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["NPROC_PER_NODE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(find_free_port())
            os.environ["NVTE_TORCH_COMPILE"] = str(0)
            mp.spawn(
                run_module_reshard,
                nprocs=world_size,
                args=(world_size, self.gt_ckpt, self.hidden_size, self.test_dir_prefix, True),
                join=True,
                daemon=False,
                start_method="spawn",
            )
            atorch.reset_distributed()

        def load(ckpt_path):
            util = ShardTensorUtil(ckpt_path, 0, 1, device="cpu")
            buffers = {i: util.load_tensor_by_name(i) for i in util.buffers.keys()}
            params = {i: util.load_tensor_by_name(i) for i in util.param_meta.keys()}
            return {**buffers, **params}

        # load flat param ckpt which generated in step 1
        # save and reshard
        shard_numbers = [2, 3, 4]
        loaded_state_dict = []
        for shard in shard_numbers:
            boot_inner(shard)
            loaded_state_dict.append(load(f"/tmp/fsdp_speedup_init_test/{self.test_dir_prefix}/reshard/{shard}"))
        gt_state_dict = load(self.gt_ckpt)
        for state_dict in loaded_state_dict:
            for name, gt_value in gt_state_dict.items():
                # only save/load, all values should be equal.
                self.assertTrue((gt_value == state_dict[name]).all())


# not a test, if nameing starts with `test`, pytest will run it.
def _test_generator(world_size, hidden):
    class TestClass(FSDPShardSaveLoadTest):
        def __init__(self, methodName="runTest"):
            super().__init__(methodName, world_size, hidden)

    TestClass.__name__ = f"FSDPShardSaveLoadTest_{world_size}_{hidden}"
    return TestClass


FSDPShardSaveLoadTest_4_64 = _test_generator(4, 64)

if torch.cuda.device_count() == 8:
    FSDPShardSaveLoadTest_7_97 = _test_generator(7, 97)

if __name__ == "__main__":
    unittest.main()
