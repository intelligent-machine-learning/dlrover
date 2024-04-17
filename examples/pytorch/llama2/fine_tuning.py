# Copyright 2023 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fire
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)

from dlrover.trainer.torch.flash_checkpoint.hf_trainer import FlashCkptTrainer

from .ascend_utils import is_torch_npu_available

CUTOFF_LEN = 512


class PrintCudaMemCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        cuda_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"cuda memory {cuda_mem:.3f}G")


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""


def tokenize(tokenizer, prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(tokenizer, data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)
    return tokenized_full_prompt


def train(data_path, model_name_or_path="meta-llama/Llama-2-7b-hf"):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
        ignore_mismatched_sizes=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=False,
        ignore_mismatched_sizes=True,
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"

    data = load_dataset("json", data_files=data_path)
    data["train"]

    train_val = data["train"].train_test_split(
        test_size=200, shuffle=True, seed=42
    )
    train_data = train_val["train"].map(
        lambda x: generate_and_tokenize_prompt(tokenizer, x)
    )

    val_data = train_val["test"].map(
        lambda x: generate_and_tokenize_prompt(tokenizer, x)
    )

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    MICRO_BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    TRAIN_STEPS = 3000
    OUTPUT_DIR = "experiments"

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        warmup_steps=100,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = FlashCkptTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator,
        callbacks=[PrintCudaMemCallback()],
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if not is_torch_npu_available():
        model = torch.compile(model)

    last_ckpt_path = trainer.get_last_checkpoint()
    trainer.train(resume_from_checkpoint=last_ckpt_path)
    model.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    fire.Fire(train)
