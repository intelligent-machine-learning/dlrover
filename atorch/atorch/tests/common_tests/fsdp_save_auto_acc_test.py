import contextlib
import itertools
import os
import unittest
from collections.abc import Mapping

import pytest

torch = pytest.importorskip("torch", "2.0.9")
import torch.multiprocessing as mp  # noqa: E402
from torch.distributed.fsdp import FullStateDictConfig  # noqa: E402
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: E402
from torch.distributed.fsdp import OptimStateKeyType, StateDictType  # noqa: E402

import atorch  # noqa: E402
from atorch.auto.accelerate import auto_accelerate  # noqa: E402
from atorch.common.util_func import find_free_port  # noqa: E402
from atorch.tests.toy_modules.toy_module import create_model_context, get_gpt2_module_type, run_train  # noqa: E402
from atorch.utils.fsdp_save_util import (  # noqa: E402
    ShardOptim,
    ShardTensorUtil,
    save_fsdp_flat_param,
    save_fsdp_optim_param,
)
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload  # noqa: E402
from atorch.utils.version import torch_version  # noqa: E402


def compare_optim(path, model, rank):
    origin = torch.load(f"{path}/origin_state.pt")
    sm = ShardOptim(path)
    reshard_optim_state = sm.reshard_optim_state_dict(model)
    optim_state_dict = origin["optimizer_state_dict"]
    origin_reshard_origin_state = FSDP.shard_full_optim_state_dict(
        optim_state_dict, model
    )  # may be removed after PyTorch 2.2
    rekeyed_reshard_origin_state = FSDP.rekey_optim_state_dict(
        origin_reshard_origin_state, OptimStateKeyType.PARAM_NAME, model
    )
    rekeyed_reshard_optim_state = FSDP.rekey_optim_state_dict(reshard_optim_state, OptimStateKeyType.PARAM_NAME, model)

    def compare(a, b, name, result):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            r = (a == b).all()
            if isinstance(r, torch.Tensor):
                r = r.item()
            result[name] = r
        elif isinstance(a, Mapping) and isinstance(b, Mapping):
            if len(a) != len(b):
                raise ValueError
            for k in a:
                compare(a[k], b[k], f"{name}-{k}", result)
        else:
            result[name] = a == b

    result = {}
    compare(rekeyed_reshard_optim_state, rekeyed_reshard_origin_state, str(rank), result)
    return all(result.values()), reshard_optim_state


def run_gpt2_with_strategy(
    rank, use_bf16, fsdp_config, ckpt_path, is_save, is_load, world_size, compare_range=None, is_train=False
):
    hidden_size = 256
    head_num = 4
    layer_num = 3
    seq_length = 512
    data_size = 16
    batch_size = 2
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fa = torch.cuda.get_device_capability()[0] >= 8

    def func(name):
        return name

    shard_kwargs = {"name_mapping_func_or_dict": func}
    ctx = (
        contextlib.nullcontext()
        if not is_load
        else init_empty_weights_with_disk_offload(
            ckpt_path=ckpt_path, shard_kwargs=shard_kwargs, meta_init_offload_name="1"
        )
    )
    with ctx:
        model_context = create_model_context(
            data_size=data_size,
            batch_size=batch_size,
            use_optim_param_func=True,
            use_gpt2=True,
            hidden_size=hidden_size,
            head_num=head_num,
            layer_num=layer_num,
            seq_length=seq_length,
        )
    ctx = (
        contextlib.nullcontext()
        if not is_load
        else init_empty_weights_with_disk_offload(
            ckpt_path=ckpt_path, shard_kwargs=shard_kwargs, meta_init_offload_name="2"
        )
    )
    with ctx:
        model_context_1 = create_model_context(
            data_size=data_size,
            batch_size=batch_size,
            use_optim_param_func=True,
            use_gpt2=True,
            hidden_size=hidden_size,
            head_num=head_num,
            layer_num=layer_num,
            seq_length=seq_length,
        )

    amp_config = {"dtype": torch.bfloat16 if use_bf16 else torch.float16}
    default_fsdp_config = {
        "limit_all_gathers": True,
    }
    default_fsdp_config.update(fsdp_config if fsdp_config is not None else {})

    if torch_version() >= (2, 0, 0):
        default_fsdp_config["use_orig_params"] = True
    checkpoint_config = default_fsdp_config["atorch_wrap_cls"]

    if use_fa:
        strategy = [
            "parallel_mode",
            "module_replace",
            ("amp_native", amp_config),
            ("fsdp", default_fsdp_config),
            ("checkpoint", checkpoint_config),
        ]
    else:
        strategy = [
            "parallel_mode",
            ("amp_native", amp_config),
            ("fsdp", default_fsdp_config),
            ("checkpoint", checkpoint_config),
        ]

    status, result, best_strategy = auto_accelerate(
        model_context.model,
        model_context.optim_func,
        model_context.dataset,
        model_context.loss_func,
        model_context.prepare_input,
        model_context.model_input_format,
        model_context.optim_args,
        model_context.optim_param_func,
        model_context.dataloader_args,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=True,
        meta_init_offload_name="1",
    )
    status, result_1, best_strategy = auto_accelerate(
        model_context_1.model,
        model_context_1.optim_func,
        model_context_1.dataset,
        model_context_1.loss_func,
        model_context_1.prepare_input,
        model_context_1.model_input_format,
        model_context_1.optim_args,
        model_context_1.optim_param_func,
        model_context_1.dataloader_args,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=True,
        meta_init_offload_name="2",
    )
    assert status
    assert len(best_strategy) == len(strategy)
    if is_load:
        for (_, p), (_, p1) in zip(result.model.named_parameters(), result_1.model.named_parameters()):
            r = p == p1
            if isinstance(r, torch.Tensor):
                r = r.all().item()
            if not r:
                raise ValueError("all auto acc parameters must be same")
    m_model = result.model
    m_dataloader = result.dataloader
    m_optim = result.optim
    m_prepare_input = result.prepare_input
    m_loss_func = result.loss_func
    input_dtype = torch.float32
    if is_load:
        reshard_optim_state = None
        for i in compare_range:
            result, reshard_optim_state = compare_optim(i, m_model, rank)
            if not result:
                raise ValueError("optim error")
        m_optim.load_state_dict(reshard_optim_state)

    if is_train:
        run_train(
            m_model,
            m_dataloader,
            m_optim,
            m_prepare_input,
            m_loss_func,
            device,
            input_dtype=input_dtype,
            gpt2_model=True,
        )
    if is_save:
        save_fsdp_flat_param(m_model, ckpt_path)
        save_fsdp_optim_param(m_model, m_optim, ckpt_path)
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(m_model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = m_model.state_dict()
            optim_state_dict = FSDP.full_optim_state_dict(m_model, m_optim)

        if rank == 0:
            torch.save(
                {
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optim_state_dict,
                },
                f"{ckpt_path}/origin_state.pt",
            )
    atorch.reset_distributed()


class FSDPShardSaveLoadTest(unittest.TestCase):
    def setUp(self):
        atorch.reset_distributed()
        self.has_test_weights = False

    def tearDown(self):
        atorch.reset_distributed()

    def _cmp_param(self, path):
        shard = ShardTensorUtil(path, 1, 256, device="cpu")
        origin = torch.load(f"{path}/origin_state.pt")
        model_state = origin["model_state_dict"]
        a = []
        for k, v in model_state.items():
            s = shard.load_tensor_by_name(k)
            v = v.to("cpu")
            if s is not None:
                a.append((s == v).all().item())
        return all(a)

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Must have at least 4 GPUs for gpu test",
    )
    def test_gpt2_with_exist_weight(self):
        if not self.has_test_weights:
            return

        self.assertTrue(self._cmp_param("/tmp/test_fsdp_save/2/block/"))
        fsdp_config = {
            "sync_module_states": True,
            "atorch_wrap_cls": tuple(get_gpt2_module_type(module=i) for i in ("mlp", "attn")),
        }
        world_size = 4
        args = (
            True,
            fsdp_config,
            "/tmp/test_fsdp_save/2/block/",
            True,
            True,
            world_size,
            ["/tmp/test_fsdp_save/2/block/"],
            True,
        )
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["NPROC_PER_NODE"] = str(world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_gpt2_with_strategy,
            args=args,
            nprocs=world_size,
            join=True,
            daemon=False,
            start_method="spawn",
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4,
        "Must have at least 4 GPUs for gpu test",
    )
    def test_gpt2_save_and_prepare(self):
        """This test will test different wrap class: block or (mlp, attt), world size (2,3,4).
        Save shard checkpoint and origin checkpoint, compare them.
        Save optim and use FSDP.shard_full_optim_state_dict to reshard optim state, compare the
        reshard optim state.
        """
        if self.has_test_weights:
            return
        gpt2block_cls = [["block"], ["mlp", "attn"]]
        world_sizes = [2, 4]
        compare_optim_paths = []
        # get all combinations
        world_sizes_and_gpt2block_cls = list(itertools.product(world_sizes, gpt2block_cls))

        def boot_inner_test(self, world_size, cls, bf16, test_optim=False):
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["NPROC_PER_NODE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(find_free_port())

            wrap_cls = tuple(get_gpt2_module_type(module=i) for i in cls)

            fsdp_config = {
                "sync_module_states": True,
                "atorch_wrap_cls": wrap_cls,
            }
            save_path = f"/tmp/test_fsdp_save/{world_size}/{'-'.join(cls)}"
            compare_optim_paths.append(save_path)
            if test_optim:
                args = (bf16, fsdp_config, compare_optim_paths[0], False, True, world_size, compare_optim_paths)
            else:
                args = (bf16, fsdp_config, save_path, True, False, world_size, None)
            mp.spawn(
                run_gpt2_with_strategy,
                args=args,
                nprocs=world_size,
                join=True,
                daemon=False,
                start_method="spawn",
            )
            if not test_optim:
                self.assertTrue(self._cmp_param(save_path))

        for world_size, cls in world_sizes_and_gpt2block_cls:
            boot_inner_test(self, world_size, cls, bf16=True, test_optim=False)
            boot_inner_test(self, world_size, cls, bf16=False, test_optim=False)
        for world_size, cls in world_sizes_and_gpt2block_cls:
            boot_inner_test(self, world_size, cls, bf16=True, test_optim=True)
            boot_inner_test(self, world_size, cls, bf16=False, test_optim=True)
        atorch.reset_distributed()


if __name__ == "__main__":
    unittest.main()
