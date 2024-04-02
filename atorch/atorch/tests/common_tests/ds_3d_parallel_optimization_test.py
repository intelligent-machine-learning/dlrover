import os
import unittest

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

import atorch
from atorch.auto import auto_accelerate
from atorch.auto.opt_lib.ds_3d_parallel_optimization import DeepSpeed3DParallelConfig
from atorch.common.util_func import find_free_port
from atorch.modules.distributed_modules.cross_entropy import vocab_parallel_cross_entropy
from atorch.tests.common_tests.ds_pipe_test import _weight_align as _ds_pipe_weight_align
from atorch.tests.common_tests.ds_pipe_test import gpt2_custom_patcher
from atorch.tests.common_tests.manual_tp_test import _weight_align as _tp_weigth_align
from atorch.tests.common_tests.manual_tp_test import get_gpt2_tpinfo
from atorch.utils.manual_tp_utils import tp_manual_shard_custom_fn
from atorch.utils.meta_model_utils import build_recorded_module, record_module_init
from atorch.utils.version import torch_version


def get_gpt2_3d_parallel_cfg(gpt2_model_config):
    def batch_fn(data):
        tokens = data["input_ids"]
        position_ids = data["position_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]
        loss_mask = data["loss_mask"]
        return (tokens, position_ids, attention_mask), (
            labels,
            loss_mask,
        )

    ds_config = {
        "train_micro_batch_size_per_gpu": 4,  # relevant to data_iter
        "gradient_accumulation_steps": 8,
        "pipeline": {"activation_checkpoint_interval": 1},
        "fp16": {"enabled": True},
    }

    ds_3d_parallel_cfg = DeepSpeed3DParallelConfig(
        tpinfo=get_gpt2_tpinfo(),
        custom_patcher=gpt2_custom_patcher(gpt2_model_config),
        ds_config=ds_config,
        batch_fn=batch_fn,
    )
    return ds_3d_parallel_cfg


def _weight_align(model, ref_model, tp_model):
    _tp_weigth_align(tp_model, ref_model)
    _ds_pipe_weight_align(model, tp_model)


def run_ds_3d_parallel(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)
    atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)

    # record meta init model
    device = torch.cuda.current_device()
    gpt2_model_config = GPT2Config(
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        n_embd=256,
        n_head=4,
        n_layer=3,
        n_positions=512,
        vocab_size=50000,
    )
    # meta model for ds 3d parallel
    with record_module_init():
        meta_model = GPT2Model(gpt2_model_config)

    strategy = [
        ("parallel_mode", ([("tensor", 2), ("data", 1), ("pipeline", 2)], None)),
        ("deepspeed_3d_parallel", get_gpt2_3d_parallel_cfg(gpt2_model_config)),
    ]

    def my_loss_func(logits, labels):
        labels, loss_mask = labels[0], labels[1]
        logits = logits.float()
        losses = vocab_parallel_cross_entropy(logits, labels).view(-1)
        loss = torch.sum(losses * loss_mask.view(-1))
        if loss_mask.sum().item() > 0:
            loss = loss / loss_mask.sum()
        return loss

    def optim_param_func(model):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 1e-1,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    status, result, best_strategy = auto_accelerate(
        meta_model,
        torch.optim.AdamW,
        optim_args={"lr": 1e-5},
        optim_param_func=optim_param_func,
        loss_func=my_loss_func,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=True,
    )
    assert status, f"auto_accelerate failed. status: {status}, result: {result}, best_strategy: {best_strategy}"
    model = result.model

    # ref model, tp model for weight align
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    ref_model = GPT2Model(gpt2_model_config).to(device)
    with record_module_init():
        tp_meta_model = GPT2Model(gpt2_model_config)
    tp_manual_shard_custom_fn(tp_meta_model, get_gpt2_tpinfo())
    tp_model = build_recorded_module(tp_meta_model).to(device)
    _weight_align(model.module, ref_model, tp_model)

    def gpt2_dummy_data_iter():
        cnt = 0
        while True:
            torch.manual_seed(cnt)
            torch.cuda.manual_seed(cnt)
            input_ids = torch.randint(0, 10000, (4, 512), device=device)
            position_ids = torch.arange(0, 512, dtype=torch.long, device=device)
            attention_mask = torch.randint(0, 2, (4, 512), device=device)
            labels = torch.randint(0, 10000, (4, 512), device=device)
            loss_mask = torch.randint(0, 2, (4, 512), device=device)
            yield locals()
            cnt += 1

    # train_batch compute
    data_iter = gpt2_dummy_data_iter()
    ds_loss = model.train_batch(data_iter)

    # ref compute
    ref_data_iter = gpt2_dummy_data_iter()
    total_loss = None
    for _ in range(8):  # relevant to gradient_accumulation_steps
        data = next(ref_data_iter)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            last_hidden_state = ref_model(
                input_ids=data["input_ids"], position_ids=data["position_ids"], attention_mask=data["attention_mask"]
            ).last_hidden_state
            logits = F.linear(last_hidden_state, ref_model.wte.weight)
            losses = torch.nn.CrossEntropyLoss(reduction="none")(
                logits.view(-1, logits.size(-1)), data["labels"].view(-1)
            )
        loss = torch.sum(losses * data["loss_mask"].view(-1))
        if data["loss_mask"].sum().item() > 0:
            loss = loss / data["loss_mask"].sum()
        if total_loss is None:
            total_loss = torch.zeros_like(loss)
        total_loss += loss.detach()
    ref_loss = total_loss / 8

    assert torch.all(torch.isclose(ds_loss, ref_loss)), f"ds_loss: {ds_loss}, ref_loss: {ref_loss}"
    atorch.reset_distributed()


class TestDeepSpeed3DParallel(unittest.TestCase):
    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch_version() < (2, 0, 0),  # type: ignore
        "Must have at least 4 GPUs for tensor + pipeline parallel test",
    )
    def test_ds_3d_parallel(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_ds_3d_parallel,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""


if __name__ == "__main__":
    unittest.main()
