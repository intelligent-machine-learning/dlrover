# Copyright 2024 The DLRover Authors. All rights reserved.
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
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizerFast,
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


def train(data_path):
    # The default model is llama-7B and we can set the model size by
    # setting hidden_size, num_attention_heads, num_hidden_layers.
    config = LlamaConfig()  # llama-7B
    model = LlamaForCausalLM(config)
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "hf-internal-testing/llama-tokenizer"
    )

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"
    data = load_dataset("json", data_files=data_path)

    train_val = data["train"].train_test_split(
        test_size=200, shuffle=True, seed=42
    )
    train_data = train_val["train"].map(
        lambda x: generate_and_tokenize_prompt(tokenizer, x)
    )

    val_data = train_val["test"].map(
        lambda x: generate_and_tokenize_prompt(tokenizer, x)
    )

    MICRO_BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    TRAIN_STEPS = 10000
    OUTPUT_DIR = "experiments"

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        warmup_steps=100,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        output_dir=OUTPUT_DIR,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="tensorboard",
        deepspeed="deepspeed_config.json",
        save_safetensors=False,
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
    if not is_torch_npu_available():
        model = torch.compile(model)
    last_ckpt_path = trainer.get_last_checkpoint()
    trainer.train(resume_from_checkpoint=last_ckpt_path)


if __name__ == "__main__":
    fire.Fire(train)
