import subprocess

import pytest
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM

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


def run_atorch_trainer(test_args):
    train_dataset = LlamaDatset(100)
    eval_dataset = LlamaDatset(20)

    atorch_opt = test_args.get("atorch_opt", "fsdp")
    save_load_by_streaming = test_args.get("save_load_by_streaming", True)
    use_atorch_dataloader = test_args.get("use_atorch_dataloader", True)
    use_default_data_collator = test_args.get("use_default_data_collator", True)
    async_save = test_args.get("async_save", False)

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
        save_load_by_streaming=save_load_by_streaming,
        save_strategy="steps",
        save_steps=0.4,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=0.9,
        logging_strategy="steps",
        logging_steps=0.1,
        logging_nan_inf_filter=False,
        gradient_checkpointing=True,
        atorch_opt=atorch_opt,
        atorch_wrap_cls=(LlamaDecoderLayer,),
        model_input_format="unpack_dict",
        use_atorch_dataloader=use_atorch_dataloader,
        use_default_data_collator=use_default_data_collator,
        async_save=async_save,
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
    assert isinstance(trainer.model, FSDP if atorch_opt == "fsdp" else DDP)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip cpu ut, only run on gpu.")
@pytest.mark.skipif(torch_version() < (2, 0, 0), reason="AtorchTrainer need torch2.0 .")
@pytest.mark.parametrize("atorch_opt", ["fsdp", "ddp"])
@pytest.mark.parametrize("save_load_by_streaming", [True, False])
@pytest.mark.parametrize("use_atorch_dataloader", [True, False])
@pytest.mark.parametrize("use_default_data_collator", [True, False])
@pytest.mark.parametrize("async_save", [True, False])
def test_atorch_trainer(
    atorch_opt, save_load_by_streaming, use_atorch_dataloader, use_default_data_collator, async_save
):
    # save_load_by_streaming only works on fsdp training mode.
    if atorch_opt == "ddp" and save_load_by_streaming:
        pytest.skip()

    # Skip some tests to reduce the time of unit test.
    if async_save:
        if save_load_by_streaming or not use_atorch_dataloader or not use_default_data_collator:
            pytest.skip()

    test_args = {
        "atorch_opt": atorch_opt,
        "save_load_by_streaming": save_load_by_streaming,
        "use_atorch_dataloader": use_atorch_dataloader,
        "use_default_data_collator": use_default_data_collator,
        "async_save": async_save,
    }

    gpu_num = 4
    dist_cmd = f"coverage run -m atorch.distributed.run --nnode=1 --nproc_per_node={gpu_num} --node_rank=0 {__file__}"

    for k, v in test_args.items():
        if isinstance(v, bool):
            if v:
                dist_cmd += f" --{k}"
        else:
            dist_cmd += f" --{k} {v}"

    subprocess.run(dist_cmd, shell=True)


if __name__ == "__main__":
    import argparse

    import coverage

    ut_cov = coverage.Coverage()
    ut_cov.start()

    parser = argparse.ArgumentParser(description="Test atorch trainer.")

    parser.add_argument("--atorch_opt", type=str, help="ATorch optimize method.")
    parser.add_argument(
        "--save_load_by_streaming", action="store_true", help="Whether to use stream saving when using FSDP."
    )
    parser.add_argument(
        "--use_atorch_dataloader", action="store_true", help="Whether to use auto_accelerate to generate dataloader."
    )
    parser.add_argument(
        "--use_default_data_collator", action="store_true", help="Whether to use default data collator in trainer."
    )
    parser.add_argument(
        "--async_save", action="store_true", help="Whether to use asynchronous saving model and optimizer."
    )

    test_args = vars(parser.parse_args())

    run_atorch_trainer(test_args)

    ut_cov.stop()
    ut_cov.save()
