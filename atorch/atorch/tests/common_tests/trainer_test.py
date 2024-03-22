import os
import unittest

import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

import atorch
from atorch.common.util_func import find_free_port
from atorch.trainer.atorch_args import AtorchArguments
from atorch.trainer.atorch_trainer import AtorchTrainer
from atorch.utils.version import torch_version


class LlamaDatset(Dataset):
    def __init__(self, size=100, max_words=30):
        self.size = 100
        self.max_words = max_words

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.ones([self.max_words], dtype=torch.int64)
        labels = torch.ones([self.max_words], dtype=torch.int64)
        attention_mask = torch.ones([self.max_words], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def run_atorch_trainer(rank, world_size):
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NPROC_PER_NODE"] = str(world_size)

    train_dataset = LlamaDatset(100)
    eval_dataset = LlamaDatset(20)

    config = LlamaConfig(
        vocab_size=10,
        hidden_size=32,
        intermediate_size=1,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=1,
    )
    model = LlamaForCausalLM(config)
    args = AtorchArguments(
        output_dir="/tmp/output_atorch_trainer",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        do_train=True,
        fp16=True,
        save_load_by_streaming=True,
        save_strategy="steps",
        save_steps=0.8,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=0.9,
        logging_strategy="steps",
        logging_steps=0.5,
        logging_nan_inf_filter=False,
        gradient_checkpointing=True,
        atorch_opt="fsdp",
        atorch_wrap_cls=(LlamaDecoderLayer,),
        model_input_format="unpack_dict",
        use_atorch_dataloader=False,
    )
    trainer = AtorchTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(torch.optim.AdamW, None),
    )
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    assert isinstance(trainer.model, FSDP)
    assert isinstance(metrics, dict)
    atorch.reset_distributed()


@unittest.skipIf(not torch.cuda.is_available(), "Skip cpu ut, only run on gpu.")
@unittest.skipIf(torch_version() < (2, 0, 0), "AtorchTrainer need torch2.0 .")
class AtorchTrainerTest(unittest.TestCase):
    def test_atorch_trainer(self):
        world_size = 4
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(
            run_atorch_trainer,
            args=(world_size,),
            nprocs=world_size,
            join=True,
        )
        os.environ["MASTER_ADDR"] = ""
        os.environ["MASTER_PORT"] = ""
