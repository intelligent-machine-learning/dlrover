import copy
import os
import unittest

import torch
import torch.multiprocessing as mp
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.auto.opt_lib.amp_optimization import is_fp8_available
from atorch.auto.opt_lib.zero_optimization import get_skip_match_module_child_wrap_policy
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import parallel_instance_index, parallel_instance_num, world_size
from atorch.optimizers import BF16Optimizer
from atorch.tests.toy_modules.toy_module import (
    create_model_context,
    get_gpt2_module_type,
    run_train,
    sp_data_process_fn,
)
from atorch.utils.version import torch_version


def run_wraps(model_context):
    gpt2block_cls = get_gpt2_module_type(module="block")
    wrap_type_tuple = (gpt2block_cls, torch.nn.LayerNorm)
    fsdp_config = {
        "sync_module_states": True,
        "limit_all_gathers": True,
        "atorch_wrap_cls": wrap_type_tuple,
    }
    strategy = [
        "parallel_mode",
        ("fsdp", fsdp_config),
    ]
    mc_copy = copy.deepcopy(model_context)
    status, result, _ = auto_accelerate(
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
    )
    assert status
    fsdp_module_counts = {t: 0 for t in wrap_type_tuple}
    for _, child in result.model.named_modules():
        if isinstance(child, FSDP):
            for t in wrap_type_tuple:
                if isinstance(child.module, t):
                    fsdp_module_counts[t] += 1
    assert fsdp_module_counts[wrap_type_tuple[0]] == 3
    assert fsdp_module_counts[wrap_type_tuple[1]] == 3 * 2 + 1

    model_context = copy.deepcopy(mc_copy)
    # test get_skip_match_module_child_wrap_policy with module class tuple
    wrap_policy = get_skip_match_module_child_wrap_policy(wrap_type_tuple)
    fsdp_config = {
        "sync_module_states": True,
        "limit_all_gathers": True,
        "auto_wrap_policy": wrap_policy,
    }
    strategy = [
        "parallel_mode",
        ("fsdp", fsdp_config),
    ]
    status, result, _ = auto_accelerate(
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
    )
    assert status
    fsdp_module_counts = {t: 0 for t in wrap_type_tuple}
    for _, child in result.model.named_modules():
        if isinstance(child, FSDP):
            for t in wrap_type_tuple:
                if isinstance(child.module, t):
                    fsdp_module_counts[t] += 1
    assert fsdp_module_counts[wrap_type_tuple[0]] == 3
    assert fsdp_module_counts[wrap_type_tuple[1]] == 1

    model_context = mc_copy
    # test get_skip_match_module_child_wrap_policy with module class and name tuple
    wrap_type_tuple_name = (gpt2block_cls, "LayerNorm")
    wrap_policy = get_skip_match_module_child_wrap_policy(wrap_type_tuple_name, model_context.model)
    fsdp_config = {
        "sync_module_states": True,
        "limit_all_gathers": True,
        "auto_wrap_policy": wrap_policy,
    }
    strategy = [
        "parallel_mode",
        ("fsdp", fsdp_config),
    ]
    status, result, _ = auto_accelerate(
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
    )
    assert status
    fsdp_module_counts = {t: 0 for t in wrap_type_tuple}
    for _, child in result.model.named_modules():
        if isinstance(child, FSDP):
            for t in wrap_type_tuple:
                if isinstance(child.module, t):
                    fsdp_module_counts[t] += 1
    assert fsdp_module_counts[wrap_type_tuple[0]] == 3
    assert fsdp_module_counts[wrap_type_tuple[1]] == 1


def run_ds_zero(
    rank,
    hidden_size,
    head_num,
    layer_num,
    seq_length,
    data_size,
    batch_size,
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")

    mc = create_model_context(
        data_size=data_size,
        batch_size=batch_size,
        use_gpt2=True,
        hidden_size=hidden_size,
        head_num=head_num,
        layer_num=layer_num,
        seq_length=seq_length,
    )

    def train_with_ds(model_context, dtype, ds_zero, cpu_offload):
        strategy = ["parallel_mode"]
        if dtype != torch.float32:
            strategy.append(("half", "fp16" if dtype == torch.half else "bf16"))

        ds_config = {"use_ds_zero": True}
        ds_config["cpu_offload"] = cpu_offload
        if ds_zero == "zero2":
            ds_config["static_loss_scale"] = 0
            strategy.append(("zero2", ds_config))
        else:
            strategy.append(("zero1", ds_config))

        data_size = len(model_context.dataset)

        status, result, _ = auto_accelerate(
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
        )
        assert status
        assert isinstance(result.optim, DeepSpeedZeroOptimizer)

        m_model = result.model
        m_dataloader = result.dataloader
        m_optim = result.optim
        m_prepare_input = result.prepare_input
        m_loss_func = result.loss_func

        device = "cuda" if torch.cuda.is_available() else "cpu"

        num = run_train(
            m_model,
            m_dataloader,
            m_optim,
            m_prepare_input,
            m_loss_func,
            device,
            gpt2_model=True,
            use_optim_backward=result.args["use_optim_backward"],
        )
        assert num == data_size // batch_size, f"num={num}"

    dtype_list = [torch.float16, torch.half, torch.float32]
    cpu_offload_list = [False, True]
    ds_zero_list = ["zero1", "zero2"]

    for dtype in dtype_list:
        for ds_zero in ds_zero_list:
            for cpu_offload in cpu_offload_list:
                mc_copy = copy.deepcopy(mc)
                train_with_ds(mc_copy, dtype, ds_zero, cpu_offload)

    atorch.reset_distributed()


def run_gpt2_with_strategy(
    rank,
    hidden_size,
    head_num,
    layer_num,
    seq_length,
    data_size,
    batch_size,
    use_bf16=False,
    use_sp=False,
    zero_size=None,
    bf16_only=False,
    wrap_test_only=False,
    use_fp8=False,
):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fa = torch.cuda.get_device_capability()[0] >= 8
    model_context = create_model_context(
        data_size=data_size,
        batch_size=batch_size,
        use_gpt2=True,
        hidden_size=hidden_size,
        head_num=head_num,
        layer_num=layer_num,
        seq_length=seq_length,
    )

    if wrap_test_only:
        run_wraps(model_context)
        atorch.reset_distributed()
        return

    if bf16_only:
        strategy = [("half", "bf16")]
    else:
        gpt2block_cls = get_gpt2_module_type(module="block")

        amp_config = {"dtype": torch.bfloat16 if use_bf16 else torch.float16}

        fsdp_config = {
            "sync_module_states": True,
            "limit_all_gathers": True,
            "atorch_wrap_cls": (gpt2block_cls,),
        }
        if torch_version() >= (2, 0, 0):
            fsdp_config["use_orig_params"] = True
        checkpoint_config = (gpt2block_cls,)

        if zero_size is not None:
            p_config = ([("data", zero_size)], None, True)
        else:
            p_config = None
        if use_fa:
            strategy = [
                ("parallel_mode", p_config),
                "module_replace",
                ("amp_native", amp_config),
                ("fsdp", fsdp_config),
                ("checkpoint", checkpoint_config),
            ]
        else:
            strategy = [
                ("parallel_mode", p_config),
                ("amp_native", amp_config),
                ("fsdp", fsdp_config),
                ("checkpoint", checkpoint_config),
            ]
        if use_fp8:
            strategy.append("fp8")
        if use_sp:
            strategy.append(("sequence_parallel", {"sp_size": 2, "batch_sp_processing_fn": sp_data_process_fn}))

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
    )
    assert status
    diff_method_num = 0
    if use_fp8 and not is_fp8_available():
        diff_method_num += 1
    if atorch.world_size() == 1 and not bf16_only:
        diff_method_num += 1
    assert len(best_strategy) == len(strategy) - diff_method_num
    if bf16_only:
        assert isinstance(result.optim, BF16Optimizer)
    else:
        assert not isinstance(result.optim, BF16Optimizer)
    m_model = result.model
    m_dataloader = result.dataloader
    m_optim = result.optim
    m_prepare_input = result.prepare_input
    m_loss_func = result.loss_func
    input_dtype = torch.float32

    num = run_train(
        m_model, m_dataloader, m_optim, m_prepare_input, m_loss_func, device, input_dtype=input_dtype, gpt2_model=True
    )
    assert num == data_size // batch_size, f"num={num}"
    if not bf16_only:
        if zero_size is None:
            assert parallel_instance_num() == 1
            assert parallel_instance_index() == 0
        else:
            assert parallel_instance_num() == world_size() // zero_size
            assert parallel_instance_index() == rank // zero_size
            assert m_model.world_size == zero_size
            assert m_model.rank == rank % zero_size

    atorch.reset_distributed()


class LoadStrategyTest(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "Must have at least 2 GPUs for gpu test",
    )
    def test_gpt2_strategy(self):
        os.environ["WORLD_SIZE"] = str(2)
        os.environ["NPROC_PER_NODE"] = str(2)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        hidden_size = 256
        head_num = 4
        layer_num = 3
        seq_length = 512
        data_size = 16
        batch_size = 2
        mp.spawn(
            run_gpt2_with_strategy,
            args=(hidden_size, head_num, layer_num, seq_length, data_size, batch_size, False),
            nprocs=2,
            join=True,
            daemon=False,
            start_method="spawn",
        )

    @unittest.skipIf(
        (not torch.cuda.is_available() or torch.cuda.device_count() < 2) or not torch.cuda.is_bf16_supported(),
        "Must have at least 2 GPUs with bf16 supported for gpu test",
    )
    def test_gpt2_strategy_bf16(self):
        os.environ["WORLD_SIZE"] = str(2)
        os.environ["NPROC_PER_NODE"] = str(2)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        hidden_size = 256
        head_num = 4
        layer_num = 3
        seq_length = 512
        data_size = 16
        batch_size = 4
        mp.spawn(
            run_gpt2_with_strategy,
            args=(hidden_size, head_num, layer_num, seq_length, data_size, batch_size, True),
            nprocs=2,
            join=True,
            daemon=False,
            start_method="spawn",
        )

    @unittest.skipIf(
        (not torch.cuda.is_available() or torch.cuda.device_count() < 4) or not torch.cuda.is_bf16_supported(),
        "Must have at least 4 GPUs for gpu test",
    )
    def test_gpt2_strategy_multi_instance(self):
        os.environ["WORLD_SIZE"] = str(4)
        os.environ["NPROC_PER_NODE"] = str(4)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        hidden_size = 256
        head_num = 4
        layer_num = 3
        seq_length = 128
        data_size = 16
        batch_size = 4
        mp.spawn(
            run_gpt2_with_strategy,
            args=(hidden_size, head_num, layer_num, seq_length, data_size, batch_size, False, 2),
            nprocs=4,
            join=True,
            daemon=False,
            start_method="spawn",
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        "Must have GPU with bf16 supported",
    )
    def test_gpt2_bf16_only(self):
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["NPROC_PER_NODE"] = str(1)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        hidden_size = 256
        head_num = 4
        layer_num = 3
        seq_length = 128
        data_size = 16
        batch_size = 4
        run_gpt2_with_strategy(
            0,
            hidden_size,
            head_num,
            layer_num,
            seq_length,
            data_size,
            batch_size,
            use_bf16=False,
            zero_size=None,
            bf16_only=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        "Must have GPU with bf16 supported",
    )
    def test_gpt2_fp8(self):
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["NPROC_PER_NODE"] = str(1)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        hidden_size = 256
        head_num = 4
        layer_num = 3
        seq_length = 128
        data_size = 16
        batch_size = 4
        run_gpt2_with_strategy(
            0,
            hidden_size,
            head_num,
            layer_num,
            seq_length,
            data_size,
            batch_size,
            use_bf16=True,
            zero_size=None,
            bf16_only=False,
            wrap_test_only=False,
            use_fp8=True,
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "Must have at least 2 GPUs for gpu test",
    )
    def test_fsdp_wraps(self):
        os.environ["WORLD_SIZE"] = str(2)
        os.environ["NPROC_PER_NODE"] = str(2)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        hidden_size = 256
        head_num = 4
        layer_num = 3
        seq_length = 512
        data_size = 16
        batch_size = 2
        mp.spawn(
            run_gpt2_with_strategy,
            args=(hidden_size, head_num, layer_num, seq_length, data_size, batch_size, False, None, False, True),
            nprocs=2,
            join=True,
            daemon=False,
            start_method="spawn",
        )

    @unittest.skipIf(
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        "Must have GPU with bf16 supported",
    )
    def test_ds_zero_bf16(self):
        os.environ["WORLD_SIZE"] = str(2)
        os.environ["NPROC_PER_NODE"] = str(2)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        hidden_size = 64
        head_num = 4
        layer_num = 2
        seq_length = 32
        data_size = 16
        batch_size = 2
        mp.spawn(
            run_ds_zero,
            args=(
                hidden_size,
                head_num,
                layer_num,
                seq_length,
                data_size,
                batch_size,
            ),
            nprocs=2,
            join=True,
            daemon=False,
            start_method="spawn",
        )

    @unittest.skipIf(
        (not torch.cuda.is_available() or torch.cuda.device_count() < 2),
        "Must have at least 2 GPUs for gpu test",
    )
    def test_gpt2_strategy_sp2(self):
        os.environ["WORLD_SIZE"] = str(2)
        os.environ["NPROC_PER_NODE"] = str(2)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        hidden_size = 256
        head_num = 4
        layer_num = 3
        seq_length = 512
        data_size = 16
        batch_size = 4
        mp.spawn(
            run_gpt2_with_strategy,
            args=(hidden_size, head_num, layer_num, seq_length, data_size, batch_size, False, True),
            nprocs=2,
            join=True,
            daemon=False,
            start_method="spawn",
        )


if __name__ == "__main__":
    unittest.main()
