# flake8: noqa: E402
import hashlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch", minversion="2.0.9")
if torch.version.git_version != "7bcf7da3a268b435777fe87c7794c382f444e86d" or not torch.cuda.is_available():
    pytest.skip("requires pytorch 2.1 stable release", allow_module_level=True)
import torch.multiprocessing as mp
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import atorch
from atorch.auto import auto_accelerate
from atorch.common.util_func import find_free_port
from atorch.utils.fsdp_init_util import FSDPCkptConfig, FSDPInitFn
from atorch.utils.fsdp_save_util import save_lora_optim_param, save_lora_param


class MockedShardTensorUtil:
    def __init__(self, *args, **kwargs):
        self.named_shape = args[0]
        self.name_fn = kwargs.get("name_mapping_func_or_dict")

    def load_tensor_by_name(self, name, **kwargs):
        n = self.name_fn(name)
        if n not in self.named_shape:
            return None
        shape = self.named_shape[n]
        return torch.ones(shape)


def get_peft(model):
    peft_task_type = TaskType.CAUSAL_LM
    peft_config = LoraConfig(
        task_type=peft_task_type,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    return get_peft_model(model, peft_config)


def optimizer_parameter_fn(model):
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def get_model(args, rank):
    config = transformers.GPT2Config()
    config.n_layer = 4
    config.vocab_size = 10
    with torch.device("meta"):
        model = transformers.GPT2LMHeadModel(config)
    return model


@patch("atorch.utils.fsdp_init_util.ShardTensorUtil", new=MockedShardTensorUtil)
def main(rank, args):

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=True)
    if not res:
        raise Exception("init failed")

    model = get_model(args, rank)
    # element 0 of tensors does not require grad and does not have a grad_fn
    # https://github.com/huggingface/peft/issues/137#issuecomment-1445912413
    model.enable_input_require_grads()

    init_fn = None
    ckpt_config = FSDPCkptConfig()
    ckpt_config.lora_prefix = "base_model.model" if args.lora else None
    ckpt_config.flat_ckpt_path = {k: v.shape for k, v in model.named_parameters()}
    ckpt_config.flat_ckpt_path.update({k: v.shape for k, v in model.named_buffers()})

    if args.lora:
        with torch.device("meta"):
            model = get_peft(model)

    if args.lora and args.restore_lora:
        ckpt_config.lora_ckpt_path = args.lora_model_path

    init_fn = FSDPInitFn(model, rank, ckpt_config, (GPT2Block,))

    x = torch.LongTensor([[1]]).cuda()
    y = torch.LongTensor([[2]]).cuda()
    mask = torch.LongTensor([[1]]).cuda()

    fsdp_config = {
        "sync_module_states": True,
        "use_orig_params": True,
        "param_init_fn": init_fn,
        "atorch_wrap_cls": {
            GPT2Block,
        },
    }

    p_mode = ([("data", atorch.world_size())], None)
    strategy = [
        ("parallel_mode", p_mode),
        # "module_replace",
        ("fsdp", fsdp_config),
        ("amp_native", {"dtype": torch.bfloat16}),
        ("checkpoint", {"wrap_class": GPT2Block, "no_reentrant": False}),
    ]
    print("Manually loaded auto acc strategy:", strategy)
    status, result, best_strategy = auto_accelerate(
        model,
        torch.optim.AdamW,
        loss_func=lambda x: x["logits"],
        optim_args={"lr": 0.001},
        optim_param_func=optimizer_parameter_fn,
        load_strategy=strategy,
    )
    assert status, "auto_accelerate failed"
    print("Best strategy is:", best_strategy)

    # Prepare everything with our `accelerator`.
    model = result.model
    optimizer = result.optim

    # load lora optim
    if args.restore_lora and args.lora_model_path is not None:
        full_osd = torch.load(f"{args.lora_model_path}/lora_optim") if rank == 0 else None
        sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
        optimizer.load_state_dict(sharded_osd)
    loss_func = result.loss_func
    rank = atorch.rank()

    model.train()
    for i in range(3):
        result = model(x, attention_mask=mask)
        logits = loss_func(result)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        loss.backward()

        if not args.restore_lora:
            optimizer.step()
        optimizer.zero_grad()

    # save lora weights and optim
    if args.lora:
        save_lora_optim_param(model, optimizer, args.lora_model_save_path)
        save_lora_param(model, args.lora_model_save_path)


class FSDPShardSaveLoadTest(unittest.TestCase):
    def _inner_boot(self, args):
        mp.spawn(
            main,
            args=(args,),
            nprocs=2,
            join=True,
            daemon=False,
            start_method="spawn",
        )

    def _md5_dir(self, path):
        hash_md5 = hashlib.md5()
        for root, dirs, files in os.walk(path):
            files.sort()
            for filename in files:
                if filename.startswith("."):
                    continue
                file_path = os.path.join(root, filename)

                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)

        return hash_md5.hexdigest()

    @unittest.skipIf(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        "Must have at least 4 GPUs for gpu test",
    )
    def test_lora_save_and_load(self):
        """
        Testing in two steps.
            1.  Load pretrain model, init lora weights, train 3 steps and save params/optim of lora to dir.
            2.  Load lora weight from 1st step, do not train, and save  params/optim of lora to new dir, compare
                md5sum of two dirs.

        """
        os.environ["WORLD_SIZE"] = "2"
        os.environ["NPROC_PER_NODE"] = "2"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        save_args = SimpleNamespace()
        save_args.pretrain_model_path = "123"
        save_args.lora = True
        save_args.restore_lora = False
        save_args.lora_model_save_path = "/tmp/test_lora_save"

        self._inner_boot(save_args)

        load_args = SimpleNamespace()
        load_args.pretrain_model_path = "123"
        load_args.lora = True
        load_args.restore_lora = True
        load_args.lora_model_save_path = "/tmp/test_lora_save_verify"
        load_args.lora_model_path = "/tmp/test_lora_save"

        self._inner_boot(load_args)

        save_md5 = self._md5_dir("/tmp/test_lora_save")
        after_load_save_md5 = self._md5_dir("/tmp/test_lora_save_verify")
        self.assertEqual(save_md5, after_load_save_md5)


if __name__ == "__main__":
    unittest.main()
