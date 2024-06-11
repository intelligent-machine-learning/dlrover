import argparse
import json
import logging
import math
import os
import random
import time
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from functools import partial
from itertools import chain
from pprint import pformat

import datasets
import matplotlib.pyplot as plt
import torch
import transformers
from datasets import load_dataset, load_from_disk  # type: ignore[attr-defined]
from instruction_dataset_utils import InstructionDataset
from matplotlib.ticker import MaxNLocator
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer  # , LlamaAttention, LlamaMLP
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import atorch
from atorch.auto import auto_accelerate
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")


logger = logging.getLogger(__name__)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -U -r requirements.txt",
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
TRAINING_ARGS_NAME = "training_args.bin"


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


def unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def honor_type(obj, generator):
    """
    Cast a generator to the same type as obj (list, tuple or namedtuple)
    """
    try:
        return type(obj)(generator)
    except TypeError:
        # Some objects may not be able to instantiate from a generator directly
        return type(obj)(*list(generator))


def recursively_apply(
    func,
    data,
    *args,
    test_type=lambda t: isinstance(t, torch.Tensor),
    error_on_other_type=False,
    **kwargs,
):
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func,
                    o,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for o in data
            ),
        )
    elif isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func,
                    v,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for k, v in data.items()
            }
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


def gather(tensor):
    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        return torch.cat(output_tensors, dim=0)

    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)


def model_parameters_num(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return all_param, trainable_params


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--not_save_model",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="A dir containing dataset with .arrow format.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="If passed, will set trust_remote_code=True when calling from_pretrained.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=0,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--total_train_batch_size",
        type=int,
        default=8,
        help="All batch size for the training dataloader. Equals to per_device_train_batch_size * world_size.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0,
        help="Clips gradient norm of an iterable of parameters.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--no_decay", nargs="*", default=["bias", "LlamaRMSNorm.weight"], help="No decay params.")
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="The number of sub-processes to use for the dataloader.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--ignore_dryrun_on_load_strategy",
        action="store_true",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=0,
        help="Log every X updates steps. Zero means do not logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"matplotlib"`, and `"all"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--logging_steps` bigger than 0."
        ),
        choices=["all", "tensorboard", "matplotlib"],
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="If passed, will set ignore_mismatched_sizes=True when calling from_pretrained.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="If passed, use fsdp",
    )
    parser.add_argument(
        "--fsdp_cpu_offload",
        action="store_true",
        help="If passed, offload model params to cpu memory.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16_amp", "fp16_amp", "bf16"],
        default="bf16_amp",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing or not.",
    )
    parser.add_argument(
        "--peft_type",
        type=str,
        default=None,
        help="Whether use peft and use what type of peft.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="Lora attention dimension.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha parameter for Lora scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The dropout probability for Lora layers.",
    )
    parser.add_argument(
        "--lora_target_modules",
        nargs="*",
        default=["q_proj", "v_proj"],
        help="The names of the modules to apply Lora to.",
    )
    parser.add_argument(
        "--peft_task_type",
        type=str,
        default=TaskType.CAUSAL_LM,
        choices=[TaskType.SEQ_CLS, TaskType.SEQ_2_SEQ_LM, TaskType.CAUSAL_LM, TaskType.TOKEN_CLS],
        help="Peft task type.",
    )
    parser.add_argument(
        "--fsdp_wrap_trainable_outmost",
        action="store_true",
        help="If fsdp would use wrap_trainable_outmost for peft model.",
    )
    parser.add_argument(
        "--random_log_n_training_samples",
        type=int,
        default=3,
        help="Log a few random samples from the training set.",
    )
    parser.add_argument(
        "--max_shard_size",
        default=None,
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`)."
            "`None` means no shard."
        ),
    )
    parser.add_argument(
        "--enable_torch_profiler",
        action="store_true",
        help="If passed, use torch.profiler.profile",
    )
    parser.add_argument(
        "--init_emtpy_offload",
        action="store_true",
        help="If passed, use init_empty_weights_with_disk_offload. Should be used when training from scratch.",
    )
    args = parser.parse_args()

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
        and args.dataset_path is None
    ):
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "txt",
            ], "`validation_file` should be a csv, json or txt file."

    return args


# for auto_accelerate
def optim_param_func(model, args):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in args.no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in args.no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


# for auto_accelerate
def my_loss_func(_, outputs):
    if isinstance(outputs, dict):
        return outputs["loss"]


# for auto_accelerate
def my_prepare_input(batch, device):
    batch = {k: v.to(device=device, non_blocking=True) for k, v in batch.items()}
    return batch


def get_dataset(args):
    raw_datasets = None
    if is_local_main_process():
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    elif args.dataset_path is not None:
        raw_datasets = load_from_disk(args.dataset_path)
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets


def get_config(args):
    config = None
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    return config


def get_tokenizer(args):
    tokenizer = None
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=not args.use_slow_tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer


def get_model(args, config):
    model = None
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    else:
        logger.info("Training new model from scratch")
        with init_empty_weights_with_disk_offload() if args.init_emtpy_offload else nullcontext():
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=args.trust_remote_code,
            )

    if args.peft_type is not None:
        peft_config = get_peft_config(args)
        logger.info(f"Load Peft {args.peft_type} model ......")
        if args.gradient_checkpointing and args.peft_type == "lora":
            # Make Lora and gradient checkpointing compatible
            # https://github.com/huggingface/peft/issues/137
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model = get_peft_model(model, peft_config)
    return model


def get_peft_config(args):
    """
    Returns:
        config(PeftConfig)
    """
    if args.peft_type == "lora":
        peft_config = LoraConfig(
            task_type=args.peft_task_type,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
    else:
        raise NotImplementedError(f"Not support {args.peft_type}")
    return peft_config


def tokenize_dataset(args, model, raw_datasets, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    with main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    return tokenized_datasets


def process_dataset(args, tokenized_datasets, tokenizer):
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
        result["labels"] = result["input_ids"].copy()
        return result

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    with main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    return lm_datasets


def compute_training_flops(
    batch_size,
    sequence_length,
    hidden_size,
    vocab_size,
    intermediate_size,
    num_layers,
    use_gradient_checkpointing=False,
    use_peft=False,
):
    """Returns:
    hardware flops
    model flops

    The source of formula:
    Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM's
    (APPENDIX: FLOATING-POINT OPERATIONS)

    Assuming that backward pass has twice FLOPs as many as forward pass. Only matrix multiplication FLOPs are computed.
    For use_peft, backward pass FLOPS is a little more than the forward pass. Assuming equal for simplicity here.
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
    factor = 2 if use_peft else 3
    if not use_gradient_checkpointing:
        return forward_flops_wo_gc * factor, forward_flops_wo_gc * factor
    else:
        return (
            num_layers * decoder_layer_forward_flops * (factor + 1) + logits_forward_flops * factor,
            forward_flops_wo_gc * factor,
        )


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    train_prefix = "train_"
    train_prefix_len = len(train_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        elif k.startswith(train_prefix):
            new_d["train/" + k[train_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    import socket
    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())


try:
    import psutil

    PSUTILS_INSTALLED = True
except ImportError:
    PSUTILS_INSTALLED = False
    pass

try:
    from pynvml.smi import nvidia_smi

    PYNAMY_INSTALLED = True
except ImportError:
    nvidia_smi = None
    PYNAMY_INSTALLED = False


class ThroughputTimer:
    def __init__(
        self,
        batch_size,
        start_step=2,
        steps_per_output=50,
        monitor_memory=False,
        logging_fn=None,
    ):
        self.start_time = 0
        self.end_time = 0
        self.started = False
        self.batch_size = 1 if batch_size is None else batch_size
        self.start_step = start_step
        self.epoch_count = 0
        self.micro_step_count = 0
        self.global_step_count = 0
        self.total_elapsed_time = 0
        self.step_elapsed_time = 0
        self.steps_per_output = steps_per_output
        self.monitor_memory = monitor_memory
        self.logging = logging_fn
        if self.logging is None:
            from atorch.common.log_utils import default_logger

            self.logging = default_logger.info
        self.initialized = False

        if self.monitor_memory and not PSUTILS_INSTALLED:
            self.logging(
                "Unable to import `psutil`, please install package by `pip install psutil`. Set monitor_memory=False"
            )
            self.monitor_memory = False
        self.nvsmi = nvidia_smi.getInstance() if PYNAMY_INSTALLED else None

    def update_epoch_count(self):
        self.epoch_count += 1
        self.micro_step_count = 0

    def _init_timer(self):
        self.initialized = True

    def start(self):
        self._init_timer()
        self.started = True
        if self.global_step_count >= self.start_step:
            torch.cuda.synchronize()
            self.start_time = time.time()
        return self.start_time

    def stop(self, global_step=False, report_speed=True):
        if not self.started:
            return
        self.started = False
        self.micro_step_count += 1
        if global_step:
            self.global_step_count += 1
        if self.start_time > 0:
            torch.cuda.synchronize()
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            self.total_elapsed_time += duration
            self.step_elapsed_time += duration

            if global_step:
                if report_speed and self.global_step_count % self.steps_per_output == 0:
                    logging_infos = (
                        f"epoch={self.epoch_count}/micro_step={self.micro_step_count}/"
                        f"global_step={self.global_step_count}, RunningAvgSamplesPerSec={self.avg_samples_per_sec()},"
                        f" CurrSamplesPerSec={self.batch_size / self.step_elapsed_time},"
                        f" MemAllocated={round(torch.cuda.memory_allocated() / 1024**3, 2)}GB,"
                        f" MaxMemAllocated={round(torch.cuda.max_memory_allocated() / 1024**3, 2)}GB"
                    )
                    if PYNAMY_INSTALLED:
                        current_node_gpu_mem = []
                        nvsmi_gpu_memory_usage = self.nvsmi.DeviceQuery("memory.used, memory.total")["gpu"]
                        for gpu_id, memory_dict in enumerate(nvsmi_gpu_memory_usage):
                            total_memory, used_memory, unit = (
                                memory_dict["fb_memory_usage"]["total"],
                                memory_dict["fb_memory_usage"]["used"],
                                memory_dict["fb_memory_usage"]["unit"],
                            )
                            current_node_gpu_mem.append(f"GPU{gpu_id}:{int(used_memory)}/{int(total_memory)}{unit}")
                        nvismi_gpu_memory_infos = ",".join(current_node_gpu_mem)
                        logging_infos += ". " + nvismi_gpu_memory_infos
                    self.logging(logging_infos)
                    if self.monitor_memory:
                        virt_mem = psutil.virtual_memory()
                        swap = psutil.swap_memory()
                        self.logging(
                            f"epoch={self.epoch_count}/micro_step={self.micro_step_count}/"
                            f"global_step={self.global_step_count} virtual_memory %: {virt_mem.percent}, "
                            f"swap_memory %: {swap.percent}"
                        )
                self.step_elapsed_time = 0
        return self.end_time

    def avg_samples_per_sec(self):
        if self.global_step_count > 0:
            total_step_offset = self.global_step_count - self.start_step
            avg_time_per_step = self.total_elapsed_time / total_step_offset
            # training samples per second
            return self.batch_size / avg_time_per_step
        return float("-inf")


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if atorch.local_rank() == 0:
        print(f"env:\n{get_pretty_env_info()}")
        print(f"args:\n{pformat(vars(args))}")

    assert args.logging_steps >= 0, f"logging_steps must bigger or equal than 0 but got {args.logging_steps}."
    with_tracking = args.logging_steps > 0 and args.output_dir is not None
    if args.report_to is not None and not with_tracking:
        logger.info(
            f"Found args.logging_steps=={args.logging_steps} and args.output_dir=={args.output_dir}."
            "args.report_to will be ignored."
        )
    if args.output_dir is not None and is_main_process() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"output_dir is {args.output_dir}")

    config = get_config(args)
    model = get_model(args, config)

    tokenizer = get_tokenizer(args)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    num_params, num_trainable_params = model_parameters_num(model)
    if atorch.local_rank() == 0:
        logger.info(
            f"Model has {num_params} parameters and {num_trainable_params} "
            f"trainable parameters({100 * num_trainable_params / num_params:.3f}%)."
        )

    if "alpaca" in args.dataset_path:
        train_dataset = InstructionDataset(
            args.dataset_path,
            tokenizer,
            partition="train",
            max_words=args.block_size,
        )
        eval_dataset = InstructionDataset(
            args.dataset_path,
            tokenizer,
            partition="eval",
            max_words=args.block_size,
        )
    else:
        raw_datasets = get_dataset(args)
        tokenized_datasets = tokenize_dataset(args, model, raw_datasets, tokenizer)
        lm_datasets = process_dataset(args, tokenized_datasets, tokenizer)
        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), args.random_log_n_training_samples):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
        pin_memory=True,
        drop_last=True,
    )

    dataloader_args = {
        "shuffle": True,
        "collate_fn": default_data_collator,
        "batch_size": args.total_train_batch_size,
        "pin_memory": True,
        "num_workers": args.dataloader_num_workers,
        "persistent_workers": args.dataloader_num_workers > 0,
    }
    strategy = [
        ("parallel_mode", ([("data", atorch.world_size())], None)),
        "module_replace",
    ]
    if args.fsdp:
        fsdp_config = {
            "sync_module_states": True,
            "atorch_wrap_cls": (LlamaDecoderLayer,),
            "limit_all_gathers": True,
            "use_orig_params": True,
            "forward_prefetch": True,
        }
        if args.peft_type is not None and args.fsdp_wrap_trainable_outmost:
            fsdp_config["use_orig_params"] = False
            fsdp_config["wrap_trainable_outmost"] = True

        if args.fsdp_cpu_offload:
            fsdp_config["cpu_offload"] = True
            fsdp_config.pop("sync_module_states", None)

        fsdp_opt = ("fsdp", fsdp_config)
        strategy.append(fsdp_opt)
    if "amp" in args.precision:
        low_precision_dtype = torch.bfloat16 if args.precision == "bf16_amp" else torch.float16
        amp_opt = ("amp_native", {"dtype": low_precision_dtype})
        strategy.append(amp_opt)
    elif args.precision == "bf16":
        strategy.append(("half", "bf16"))
    if args.gradient_checkpointing:
        strategy.append(("checkpoint", (LlamaDecoderLayer,)))
    status, result, best_strategy = auto_accelerate(
        model,
        torch.optim.AdamW,
        train_dataset,
        loss_func=my_loss_func,
        prepare_input=my_prepare_input,
        model_input_format="unpack_dict",
        optim_args={"lr": args.learning_rate},
        optim_param_func=partial(optim_param_func, args=args),
        dataloader_args=dataloader_args,
        load_strategy=strategy,
        ignore_dryrun_on_load_strategy=args.ignore_dryrun_on_load_strategy,
        save_strategy_to_file=os.path.join(args.output_dir, "atorch_auto_acc_strategy.txt"),
        sampler_seed=args.seed,
    )
    assert status, "auto_accelerate failed"
    logger.info(f"Best strategy is: {best_strategy}")
    model = result.model
    optimizer = result.optim
    train_dataloader = result.dataloader
    loss_func = result.loss_func
    prepare_input = result.prepare_input

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    warmup_steps = 0
    num_training_steps = args.max_train_steps * args.gradient_accumulation_steps
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps * args.gradient_accumulation_steps
    elif args.warmup_ratio > 0.0:
        warmup_steps = int(num_training_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # create a summary_writer to write training metrics to tensorboard
    summary_writer = None
    report_to_tb = with_tracking and args.report_to in ("tensorboard", "all")
    if report_to_tb:
        tb_path = os.path.join(args.output_dir, default_logdir())
        summary_writer = SummaryWriter(tb_path)
        logger.info(f"Tensorboard eventfiles will be saved at {tb_path}")

    # Train!
    device = torch.device("cuda:{}".format(atorch.local_rank()))
    total_batch_size = args.total_train_batch_size * args.gradient_accumulation_steps

    if args.total_train_batch_size > 0:
        per_device_train_batch_size = int(args.total_train_batch_size / atorch.world_size())
        total_train_batch_size = args.total_train_batch_size
    elif args.per_device_train_batch_size > 0:
        per_device_train_batch_size = args.per_device_train_batch_size
        total_train_batch_size = per_device_train_batch_size * atorch.world_size()
    else:
        raise ValueError(f"per_device_train_batch_size must greater than 0 but got {per_device_train_batch_size}")

    flops_per_gpu_per_iteration, _ = compute_training_flops(
        per_device_train_batch_size,
        args.block_size,
        config.hidden_size,
        config.vocab_size,
        config.intermediate_size,
        config.num_hidden_layers,
        args.gradient_checkpointing,
        args.peft_type is not None,
    )
    tput_timer = ThroughputTimer(total_train_batch_size, start_step=2, steps_per_output=50)
    if args.enable_torch_profiler:
        profile = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.output_dir, "torch_profile")),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=False,
            with_stack=True,
            with_modules=True,
            record_shapes=True,
        )
    else:
        profile = nullcontext()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_local_main_process())
    completed_steps = 0
    completed_eval_steps = 0
    starting_epoch = 0

    # # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    total_train_losses = [[], []]  # steps, loss
    total_eval_losses = [[], []]  # steps, loss
    all_results = {}
    training_time = 0
    for epoch in range(starting_epoch, args.num_train_epochs):

        if atorch.world_size() > 1:
            train_dataloader.sampler.set_epoch(epoch)
        model.train()
        if with_tracking:
            total_loss = torch.tensor(0.0, device=model.device)
        torch.cuda.synchronize()

        current_epoch_start_time = time.time()
        with profile as prof:
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        if step % args.gradient_accumulation_steps == 0:
                            progress_bar.update(1)
                            completed_steps += 1
                        continue
                step_start_timestamp = tput_timer.start()
                batch = prepare_input(batch, device)
                outputs = model(**batch)
                loss = loss_func(None, outputs)
                # We keep track of the loss at each epoch
                if with_tracking:
                    with torch.no_grad():
                        total_loss += loss
                loss.backward()
                if args.max_grad_norm > 0:
                    if args.precision == "fp16_amp":
                        optimizer.unscale_()
                    if isinstance(model, FSDP):
                        total_grad_norm = model.clip_grad_norm_(args.max_grad_norm)
                    else:
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if prof is not None and isinstance(prof, torch.profiler.profile):
                    prof.step()

                step_stop_timestamp = tput_timer.stop(global_step=True, report_speed=is_local_main_process())
                current_step_elapsed_time = step_stop_timestamp - step_start_timestamp
                if current_step_elapsed_time > 0:
                    flops_per_sec = flops_per_gpu_per_iteration / current_step_elapsed_time
                else:
                    flops_per_sec = 0
                progress_bar.update(1)
                completed_steps += 1

                if with_tracking and completed_steps % args.logging_steps == 0:
                    with torch.no_grad():
                        torch.distributed.all_reduce(total_loss)
                        total_loss = total_loss / atorch.world_size() / args.logging_steps
                        total_loss_cpu_value = total_loss.cpu().item()
                        total_train_losses[0].append(completed_steps)
                        total_train_losses[1].append(total_loss_cpu_value)
                        learning_rate = lr_scheduler.get_last_lr()[0]
                        tflops_per_sec = int(flops_per_sec / 1e12)
                        train_logs = {
                            "train_loss": total_loss_cpu_value,
                            "learning_rate": learning_rate,
                            "epoch": epoch,
                            "steps": completed_steps,
                            "TFLOPS": tflops_per_sec,
                        }
                        if args.max_grad_norm > 0:
                            train_logs["total_grad_norm"] = total_grad_norm.cpu().item()
                        if is_local_main_process():
                            logger.info(train_logs)
                            if report_to_tb and is_main_process():
                                train_logs = rewrite_logs(train_logs)
                                for key, value in train_logs.items():
                                    if key != "steps":
                                        summary_writer.add_scalar(f"{key}", value, completed_steps)
                        total_loss.zero_()

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                if completed_steps >= args.max_train_steps:
                    break

        torch.cuda.synchronize()
        current_epoch_elapse_time = time.time() - current_epoch_start_time
        if is_main_process():
            logger.info(f"Training epoch {epoch} takes {current_epoch_elapse_time:.3f} seconds.")
        training_time += current_epoch_elapse_time
        tput_timer.update_epoch_count()
        model.eval()
        eval_losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                batch = prepare_input(batch, device)
                outputs = model(**batch)

            loss = outputs.loss
            completed_eval_steps += 1
            gathered_loss = gather(loss.repeat(args.per_device_eval_batch_size))
            eval_losses.append(gathered_loss)
            if with_tracking:
                current_eval_step_aggregated_loss = torch.mean(gathered_loss)
                total_eval_losses[0].append(completed_eval_steps)
                total_eval_losses[1].append(current_eval_step_aggregated_loss.cpu().item())
                eval_logs = {
                    "eval_loss": current_eval_step_aggregated_loss,
                    "epoch": epoch,
                }
                if report_to_tb and is_main_process():
                    eval_logs = rewrite_logs(eval_logs)
                    for key, value in eval_logs.items():
                        summary_writer.add_scalar(f"{key}", value, global_step=completed_steps)

        eval_losses = torch.cat(eval_losses)
        try:
            eval_loss = torch.mean(eval_losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        if is_local_main_process():
            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        all_results[f"epoch{epoch}"] = {"eval_loss": eval_loss.cpu().item(), "perplexity": perplexity}

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)

    if with_tracking and args.report_to in ("all", "matplotlib") and is_main_process():
        fig, ax = plt.subplots(nrows=2, layout="constrained")
        ax[0].plot(total_train_losses[0], total_train_losses[1], label="train_loss")
        ax[0].set_xlabel("train_steps")
        ax[0].set_ylabel("train_loss")
        ax[0].set_title("Llama-2 train loss")
        ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[1].plot(total_eval_losses[0], total_eval_losses[1], label="eval_loss")
        ax[1].set_xlabel("eval_steps")
        ax[1].set_ylabel("eval_loss")
        ax[1].set_title("Llama-2 eval loss")
        ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        if args.output_dir is not None:
            fig_path = os.path.join(args.output_dir, "llama2_loss.png")
            fig.savefig(fig_path)
            logger.info(f"Loss curve has been saved at {fig_path}")

    if is_main_process():
        print(
            "Training throughput is {:.3f} samples/s".format(
                args.max_train_steps * args.total_train_batch_size / training_time,
            )
        )

    if args.output_dir is not None:
        wait_for_everyone()
        if not args.not_save_model:
            model_state_dict = None
            if isinstance(model, FSDP):
                fsdp_save_policy = FullStateDictConfig(
                    offload_to_cpu=atorch.world_size() > 1, rank0_only=atorch.world_size() > 1
                )
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fsdp_save_policy):
                    model_state_dict = model.state_dict()

            unwrapped_model = unwrap_model(model)

            if args.max_shard_size is None:
                max_shard_size_gb = math.ceil(4 * num_params / 1e9) + 1
                max_shard_size = f"{max_shard_size_gb}GB"
            else:
                max_shard_size = args.max_shard_size
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=is_main_process(),
                state_dict=model_state_dict,
                save_function=torch.save,
                max_shard_size=max_shard_size,
            )

        if is_main_process():
            tokenizer.save_pretrained(args.output_dir)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

            torch.save(args, os.path.join(args.output_dir, TRAINING_ARGS_NAME))
            logger.info(f"Ckpts and other configs have been saved at {args.output_dir}")


if __name__ == "__main__":
    status = atorch.init_distributed("nccl", set_cuda_device_using_local_rank=True)
    assert status is True, "Initialize atorch distributed context failed."
    main()
    atorch.reset_distributed()
