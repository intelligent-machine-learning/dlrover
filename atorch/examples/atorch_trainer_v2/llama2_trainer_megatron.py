#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Union

import torch
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser, TrainerCallback
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from atorch.common.log_utils import default_logger as logger
from atorch.trainer import AtorchTrainerV2, AtorchTrainingArgs
from atorch.trainer.megatron import MegatronTrainStep
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    import megatron.legacy.model
    from megatron.core import mpu
    from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
    from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.spec_utils import import_module
    from megatron.legacy.data.data_samplers import MegatronPretrainingRandomSampler, MegatronPretrainingSampler
    from megatron.training import get_args, get_tokenizer, print_rank_0
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.utils import (
        average_losses_across_data_parallel_group,
        get_batch_on_this_cp_rank,
        get_batch_on_this_tp_rank,
    )
    from megatron.training.yaml_arguments import core_transformer_config_from_yaml

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    vocab_file: Optional[str] = field(default=None, metadata={"help": "The vocab file (a json file)."})
    merge_file: Optional[str] = field(default=None, metadata={"help": "The merge file (a json file)."})
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    tokenizer_model: Optional[str] = field(default=None, metadata={"help": "The path to tokenizer model."})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the "
                "pretrained weights are loaded. set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    data_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data path (preprocessed Megatron data)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.data_path is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0("building GPT model ...")
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if use_te:
                transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                    args.num_experts, args.moe_grouped_gemm
                )
            else:
                transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )
    else:
        assert args.context_parallel_size == 1, "Context parallelism is only supported with Megatron Core!"

        model = megatron.legacy.model.GPTModel(
            config, num_tokentypes=0, parallel_output=True, pre_process=pre_process, post_process=post_process
        )

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """

    def is_dataset_built_on_rank():
        return (
            mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
        ) and mpu.get_tensor_model_parallel_rank() == 0

    def core_gpt_dataset_config_from_args(args):
        tokenizer = get_tokenizer()

        return GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=args.data_path,
            blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
            split=args.split,
            path_to_cache=args.data_cache_path,
            mock=args.mock_data,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=tokenizer,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )

    args = get_args()
    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def build_train_valid_test_data_iterators(build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    def cyclic_iter(iter):
        while True:
            for x in iter:
                yield x

    def get_train_valid_test_num_samples():
        """Train/valid/test num samples."""

        args = get_args()

        # Number of train/valid/test samples.
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters

        return (
            train_samples,
            eval_iters * args.global_batch_size,
            test_iters * args.global_batch_size,
        )

    def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
        """Build pretraining datasets."""
        train_valid_test_num_samples = get_train_valid_test_num_samples()
        print_rank_0(" > datasets target sizes (minimum size):")
        print_rank_0("    train:      {}".format(train_valid_test_num_samples[0]))
        print_rank_0("    validation: {}".format(train_valid_test_num_samples[1]))
        print_rank_0("    test:       {}".format(train_valid_test_num_samples[2]))
        return build_train_valid_test_datasets_provider(train_valid_test_num_samples)

    def build_pretraining_data_loader(dataset, consumed_samples):
        """Build dataloader given an input dataset."""

        if dataset is None:
            return None
        args = get_args()

        # Megatron sampler
        if args.dataloader_type == "single":
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
            )
        elif args.dataloader_type == "cyclic":
            batch_sampler = MegatronPretrainingRandomSampler(
                dataset,
                total_samples=len(dataset),
                consumed_samples=consumed_samples,
                micro_batch_size=args.micro_batch_size,
                data_parallel_rank=mpu.get_data_parallel_rank(),
                data_parallel_size=mpu.get_data_parallel_world_size(),
                data_sharding=args.data_sharding,
            )
        elif args.dataloader_type == "external":
            # External dataloaders are passed through. User is expected to provide a
            # torch-compatible dataloader and define samplers, if needed.
            return dataset
        else:
            raise Exception("{} dataloader type is not supported.".format(args.dataloader_type))

        # Torch dataloader.
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
        )

    def build_train_valid_test_data_loaders(build_train_valid_test_datasets_provider):
        """Build pretraining data loaders."""

        args = get_args()

        (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

        print_rank_0("> building train, validation, and test datasets ...")

        # Backward compatibility, assume fixed batch size.
        if args.iteration > 0 and args.consumed_train_samples == 0:
            assert args.train_samples is None, "only backward compatiblity support for iteration-based training"
            args.consumed_train_samples = args.iteration * args.global_batch_size
        if args.iteration > 0 and args.consumed_valid_samples == 0:
            if args.train_samples is None:
                args.consumed_valid_samples = (
                    (args.iteration // args.eval_interval) * args.eval_iters * args.global_batch_size
                )

        # Rely on distributed-aware core datasets, temporary
        is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

        # Construct the data pipeline
        if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:

            # Build datasets.
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(build_train_valid_test_datasets_provider)
            # Build dataloders.
            train_dataloader = build_pretraining_data_loader(train_ds, args.consumed_train_samples)
            if args.skip_train:
                valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
            else:
                valid_dataloader = build_pretraining_data_loader(valid_ds, args.consumed_valid_samples)
            test_dataloader = build_pretraining_data_loader(test_ds, 0)

            # Flags to know if we need to do training/validation/testing.
            do_train = train_dataloader is not None and args.train_iters > 0
            do_valid = valid_dataloader is not None and args.eval_iters > 0
            do_test = test_dataloader is not None and args.eval_iters > 0
            flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda")
        else:
            flags = torch.tensor([0, 0, 0], dtype=torch.long, device="cuda")

        torch.distributed.broadcast(flags, 0)

        args.do_train = getattr(args, "do_train", False) or flags[0].item()
        args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
        args.do_test = getattr(args, "do_test", False) or flags[2].item()

        return train_dataloader, valid_dataloader, test_dataloader

    args = get_args()

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider
    )

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ["single", "cyclic", "external"]

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return iter(dataloader)
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            return dataloader
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


class GPTTrainStep(MegatronTrainStep):
    """
    GPT train step

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args, **kwargs):
        super().__init__()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

            self.model_output_class = CausalLMOutputWithCrossAttentions

        ##################################
        # Just for testing spike loss
        self.last_loss = None
        # Just for testing spike loss
        ##################################

    def get_batch_func(self, **kwargs):
        def get_batch(data_iterator):
            """Generate a batch."""

            # TODO: this is pretty hacky, find a better way
            if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
                return None, None, None, None, None

            # get batches based on the TP rank you are on
            batch = get_batch_on_this_tp_rank(data_iterator)

            # slice batch along sequence dimension for context parallelism
            batch = get_batch_on_this_cp_rank(batch)

            return batch.values()

        return get_batch

    def get_loss_func(self, **kwargs):
        def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
            """Loss function.

            Args:
                loss_mask (torch.Tensor): Used to mask out some portions of the loss
                output_tensor (torch.Tensor): The tensor with the losses
            """
            args = get_args()

            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            if args.context_parallel_size > 1:
                loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
                torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
                loss = loss[0] / loss[1]
            else:
                loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Check individual rank losses are not NaN prior to DP all-reduce.
            if args.check_for_nan_in_loss_and_grad:
                global_rank = torch.distributed.get_rank()
                assert not loss.isnan(), (
                    f"Rank {global_rank}: found NaN in local forward loss calculation. "
                    f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
                )

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss * args.context_parallel_size, {"lm loss": averaged_loss[0]}

        return loss_func

    def get_forward_step_func(self, **kwargs):
        def forward_step(data_iterator, model: GPTModel):
            """Forward training step.

            Args:
                data_iterator : Input data iterator
                model (GPTModel): The GPT Model
            """
            # Get the batch.
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch_func()(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor, partial(self.get_loss_func(), loss_mask)

        return forward_step

    def loss_postprocessing(self, losses_reduced):
        # return super().loss_postprocessing(losses_reduced)
        """
        Loss postprocessing. Average losses across all micro-batches.

        Args:
            losses_reduced: (List[torch.Tensor]):
                A list of losses with a length of pipeline depth, which is equal to
                `global_batch_size/data_parallel_size/micro_batch_size`.
        Returns:
            A 2-tuple, the first element is a loss dict to log, and the second element is
            a ratio if the spike loss condition is met; otherwise, it will be None.

        """
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            # Average loss across microbatches.
            loss_reduced = {}
            for key in losses_reduced[0]:
                losses_reduced_for_key = [x[key] for x in losses_reduced]
                loss_reduced[key] = sum(losses_reduced_for_key) / len(losses_reduced_for_key)
            ##################################
            # Just for testing spike loss
            ratio = None
            if (
                self.last_loss is not None
                and loss_reduced["lm loss"] > self.last_loss
                and torch.abs(loss_reduced["lm loss"] - self.last_loss) > 1
            ):
                ratio = 0.8
                spike_loss = loss_reduced["lm loss"] * ratio
                loss_reduced.update(spike_loss=spike_loss)
                logger.info(
                    f' current loss {loss_reduced["lm loss"]}, last loss: {self.last_loss}, spike_loss: {spike_loss}'
                )
            self.last_loss = loss_reduced["lm loss"]
            return loss_reduced, ratio
            # Just for testing spike loss
            ##################################
            return loss_reduced, None
        return {}, None


class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass
        # print_rank_last(f"---> callback: {logs}")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AtorchTrainingArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"Training/evaluation parameters {training_args}")

    # # TODO: remove this line
    # training_args: AtorchTrainingArgs = training_args.clone()

    train_valid_test_datasets_provider.is_distributed = True

    megatron_args = dict(
        # Custom function
        # custom_prepare_model_function=None,
        custom_model_provider_function=model_provider,
        custom_megatron_dataloaders_provider_function=partial(
            build_train_valid_test_data_iterators, train_valid_test_datasets_provider
        ),
        # custom_megatron_datasets_provider_function=None,
        custom_train_step_class=GPTTrainStep,
        custom_train_step_kwargs={},
        # model args
        model_type_name="llama",  # "gpt"
        num_layers=32,  # 12
        hidden_size=4096,  # 768
        ffn_hidden_size=11008,  # 4*h
        num_attention_heads=32,  # 12
        group_query_attention=True,
        num_query_groups=32,
        max_position_embeddings=4096,
        # orig_vocab_size=model.config.vocab_size,
        position_embedding_type="rope",
        make_vocab_size_divisible_by=1,
        norm_epsilon=1e-5,
        normalization="RMSNorm",
        untie_embeddings_and_output_weights=True,
        use_flash_attn=True,
        # tokenizer
        tokenizer_type="Llama2Tokenizer",  # "GPT2BPETokenizer"
        tokenizer_model=model_args.tokenizer_model,  # For llama2
        # vocab_file=model_args.vocab_file,
        # merge_file=model_args.merge_file,
        # optimizer
        optimizer="adam",
        # Regular args
        attention_dropout=0.0,
        hidden_dropout=0.0,
        weight_decay=1e-1,
        clip_grad=1.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        # Megatron training args
        pretraining_flag=False,
        use_mcore_models=True,
        transformer_impl="transformer_engine",
        micro_batch_size=2,
        global_batch_size=16,
        add_bias_linear=False,
        bias_gelu_fusion=False,
        recompute_activations=True,
        recompute_granularity="selective",
        train_iters=10000,
        eval_iters=50,
        # Distributed args
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        sequence_parallel=True,
        distributed_backend="nccl",
        use_distributed_optimizer=True,
        # Logging args
        log_timers_to_tensorboard=True,
        log_validation_ppl_to_tensorboard=True,
        log_memory_to_tensorboard=True,
        log_throughput=True,
        log_params_norm=True,
        tensorboard_dir=training_args.tensorboard_dir,
        # Initialization args
        seed=1403,
        init_method_std=0.02,
        # Learning rate args
        lr=3e-5,
        min_lr=3e-6,
        lr_ecay_style="cosine",
        lr_warmup_fraction=0.1,
        # Mixed precision args
        # you don't need to pass in this two, since it could auto-mapping into extra configs.
        #     bf16=training_args.bf16,
        #     fp16=training_args.fp16,
        # Data
        data_path=[data_args.data_path],
        split="949,50,1",
        data_cache_path=os.path.join(training_args.output_dir, "data_cache"),
        # mock_data=True,
        seq_length=4096,
        num_workers=0,
        gradient_accumulation_fusion=False,
    )

    training_args.extra_configs = megatron_args

    # Initialize our Trainer
    trainer = AtorchTrainerV2(
        args=training_args,
        # datasets=[train_dataset, eval_dataset],
        # tokenizer=tokenizer,
        # # Data collator will default to DataCollatorWithPadding, so we change it.
        # data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics  # noqa F401

        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)
        # trainer.save_state()

    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     metrics = trainer.evaluate()

    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #     try:
    #         perplexity = math.exp(metrics["eval_loss"])
    #     except OverflowError:
    #         perplexity = float("inf")
    #     metrics["perplexity"] = perplexity

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
