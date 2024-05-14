import os
import time
from contextlib import contextmanager
from itertools import chain

import torch
from datasets import DatasetDict, load_dataset, load_from_disk  # type: ignore[attr-defined]
from deepspeed.utils import RepeatingLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import parallel_group_size, parallel_rank


def is_main_process():
    return atorch.rank() == 0


def is_local_main_process():
    return atorch.local_rank() == 0


def wait_for_everyone():
    torch.distributed.barrier()


def _goes_first(is_main):
    if is_main is False:
        wait_for_everyone()
    yield
    if is_main is True:
        wait_for_everyone()


@contextmanager
def main_process_first():
    yield from _goes_first(is_main_process())


def get_data_iter(dataset_path, tokenizer, block_size, train_micro_batch_size_per_gpu, pre_shift=True):
    if os.path.exists(dataset_path):
        raw_datasets = load_from_disk(dataset_path)
    else:
        wiki_suffix = os.path.basename(os.path.normpath(dataset_path))
        raw_datasets = DatasetDict()
        try:
            raw_datasets["train"] = load_dataset("wikitext", wiki_suffix, split="train")
            raw_datasets["validation"] = load_dataset("wikitext", wiki_suffix, split="validation")
            raw_datasets["test"] = load_dataset("wikitext", wiki_suffix, split="test")
        except Exception as e:
            logger.error(e)
            try:
                from modelscope.msdatasets import MsDataset
            except ModuleNotFoundError:
                logger.error(
                    "Can not download dataset from huggingface directyly, "
                    "you can install modelscope to download from an alternate site."
                )

            raw_datasets["train"] = MsDataset.load(
                "modelscope/wikitext", subset_name=wiki_suffix, split="train"
            ).to_torch_dataset()
            raw_datasets["validation"] = MsDataset.load(
                "modelscope/wikitext", subset_name=wiki_suffix, split="validation"
            ).to_torch_dataset()
            raw_datasets["test"] = MsDataset.load(
                "modelscope/wikitext", subset_name=wiki_suffix, split="test"
            ).to_torch_dataset()

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with main_process_first():
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=column_names)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        if pre_shift:
            # shift labels
            _labels = result["input_ids"].copy()
            result["labels"] = [each_label[1:] + each_label[:1] for each_label in _labels]

            # loss_mask drop last token
            _loss_mask = result["attention_mask"].copy()
            result["loss_mask"] = [each_loss_mask[:-1] + [0] for each_loss_mask in _loss_mask]
        else:
            result["labels"] = result["input_ids"].copy()

        # position_ids
        result["position_ids"] = [[i for i in range(block_size)] for _ in range(len(result["input_ids"]))]
        return result

    with main_process_first():
        lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    train_dataset = lm_datasets["train"]
    # eval_dataset = lm_datasets["validation"]

    sampler = DistributedSampler(
        train_dataset, num_replicas=parallel_group_size("data"), rank=parallel_rank("data"), shuffle=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        drop_last=True,
        batch_size=train_micro_batch_size_per_gpu,
        collate_fn=default_data_collator,
    )

    return RepeatingLoader(train_dataloader)


def report_memory(name):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {:.1f}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {:.1f}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {:.1f}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {:.1f}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    if torch.distributed.get_rank() == 0:
        # remove subprocess nvidia-smi call because of too slow
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def compute_llama2_training_flops(
    batch_size,
    sequence_length,
    hidden_size,
    vocab_size,
    intermediate_size,
    num_layers,
    use_gradient_checkpointing=False,
    use_lora=False,
):
    """Returns:
    hardware flops
    model flops

    The source of formula:
    Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM's
    (APPENDIX: FLOATING-POINT OPERATIONS)

    Assuming that backward pass has twice FLOPs as many as forward pass. Only matrix multiplication FLOPs are computed.
    Note: for LoRA, assuming bw pass has the same FLOPs as fw.
    """
    attention_forward_flops = (
        8 * batch_size * sequence_length * hidden_size**2 + 4 * batch_size * sequence_length**2 * hidden_size
    )
    # llama2 use gate_proj, has 3 Linears
    two_mlps_forward_flops = 3 * 2 * batch_size * sequence_length * hidden_size * intermediate_size
    logits_forward_flops = 2 * batch_size * sequence_length * hidden_size * vocab_size
    decoder_layer_forward_flops = attention_forward_flops + two_mlps_forward_flops
    # forward FLOPs without gradient checkpointing
    forward_flops_wo_gc = num_layers * decoder_layer_forward_flops + logits_forward_flops
    mul_factor = 2 if use_lora else 3
    if not use_gradient_checkpointing:
        return forward_flops_wo_gc * mul_factor, forward_flops_wo_gc * mul_factor
    else:
        return (
            num_layers * decoder_layer_forward_flops * (mul_factor + 1) + logits_forward_flops * mul_factor,
            forward_flops_wo_gc * mul_factor,
        )


def sync_and_time():
    torch.cuda.synchronize()
    return time.time()


def print_rank_0(*args, **kwargs):
    if torch.distributed.get_rank() == 0:
        print(*args, **kwargs)
