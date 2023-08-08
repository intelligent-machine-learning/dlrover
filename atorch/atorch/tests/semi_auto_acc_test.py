import os
import unittest

import torch
import torch.multiprocessing as mp

import atorch
from atorch.auto.accelerate import auto_accelerate
from atorch.common.util_func import find_free_port
from atorch.distributed.distributed import parallel_instance_index, parallel_instance_num, world_size
from atorch.tests.toy_module import create_model_context, get_gpt2_module_type, run_train
from atorch.utils.version import torch_version


def run_gpt2_with_strategy(
    rank, hidden_size, head_num, layer_num, seq_length, data_size, batch_size, use_bf16=False, zero_size=None
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
    assert len(best_strategy) == len(strategy)
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
            args=(hidden_size, head_num, layer_num, seq_length, data_size, batch_size, False, 4),
            nprocs=4,
            join=True,
            daemon=False,
            start_method="spawn",
        )


if __name__ == "__main__":
    unittest.main()
