"""
We implement the trainer to be compatible with Transformers Trainer, and many codes are taken from Transformers Trainer
with some modifications.
"""

import contextlib
import datetime
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import psutil
import safetensors
import torch
import torch.distributed as dist
from accelerate import Accelerator, skip_first_batches
from packaging import version
from peft.peft_model import PeftConfig, PeftModel
from torch import nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.multiprocessing import Manager, Pipe, Process
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator

from atorch.auto import auto_accelerate
from atorch.auto.accelerate import get_strategy
from atorch.auto.strategy import Strategy
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import is_distributed
from atorch.trainer.atorch_args import AtorchArguments
from atorch.trainer.trainer_callback import FlowCallback
from atorch.utils.fsdp_init_util import FSDPCkptConfig, FSDPInitFn
from atorch.utils.fsdp_save_util import (
    ShardOptim,
    save_fsdp_flat_param,
    save_fsdp_optim_param,
    save_lora_optim_param,
    save_lora_param,
)
from atorch.utils.hooks import ATorchHooks
from atorch.utils.import_util import is_torch_npu_available
from atorch.utils.trainer_utils import AsyncCheckpointSignal, PipeMessageEntity, get_scheduler
from atorch.utils.version import package_version_bigger_than, torch_version

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import TensorBoardCallback

# isort: on
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import OPTIMIZER_NAME, SCALER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_callback import (
    CallbackHandler,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    LengthGroupedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
    EvalPrediction,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainOutput,
    denumpify_detensorize,
    get_last_checkpoint,
    has_length,
    seed_worker,
    speed_metrics,
)
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    is_datasets_available,
    is_peft_available,
)

if TYPE_CHECKING:
    import optuna

DEFAULT_CALLBACKS = [FlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

HYPER_PARAMETER_NAME = "hyper_parameters.json"
PEFT_PARAM_PREFIX = "base_model.model."
LORA_KEY = "lora"
STREAMING_CKPT_DIR = "streaming_ckpt"  # The weight directory of Atorch FSDP model
ATORCH_LORA_WEIGHT_NAME = "lora_weight"  # The lora weight of Atorch FSDP+LORA model
ATORCH_LORA_OPTIMIZER_NAME = "lora_optim"  # The lora optimizer of Atorch FSDP+LORA model

additional_tensorboard_hook = ATorchHooks.hooks.get(ATorchHooks.ADDITIONAL_TENSORBOARD_HOOK)


def count_model_params(model):
    trainable_params = 0
    all_params = 0
    for param in model.parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    return all_params, trainable_params


class AtorchTrainer:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: AtorchArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        load_strategy: Optional[Union[str, bytes, Strategy, List]] = None,
        **kwargs,
    ):
        if args is None:
            output_dir = "tmp_atorch_trainer"
            logger.info(f"No `AtorchArguments` passed, using `output_dir={output_dir}`.")
            args = AtorchArguments(output_dir=output_dir)  # type: ignore[call-arg]

        if (
            (hasattr(args, "sharded_ddp") and len(args.sharded_ddp) > 0)
            or len(args.fsdp) > 0
            or args.deepspeed is not None
        ):
            logger.warning(
                "--sharded_ddp, --fsdp, --deepspeed in `TrainingArguments` is invalid when using `AtorchArguments`."
            )

        self.args = args
        self.kwargs = kwargs
        self.model = model

        self.atorch_fsdp = False
        if args.atorch_opt == "fsdp":
            if torch_version() < (2, 0, 0):  # type: ignore
                raise ValueError("FSDP is not supported in version of torch below 2.0")
            elif args.world_size > 1:
                self.atorch_fsdp = True
            else:
                logger.warning("FSDP will not be used if the world size is smaller or equal than 1.")

        if (self.args.load_by_streaming or self.args.save_by_streaming) and not self.atorch_fsdp:
            logger.warning("`load_by_streaming` or `save_by_streaming` is only effective when training with FSDP.")
            self.args.load_by_streaming = False
            self.args.save_by_streaming = False

        self.is_in_train = False

        # create accelerator object
        self.accelerator = Accelerator()

        # force device and distributed setup init explicitly
        args._setup_devices

        self.data_collator = None
        if data_collator is not None:
            self.data_collator = data_collator
        elif args.use_default_data_collator:
            self.data_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        if self.compute_metrics is not None and args.eval_accumulation_steps is None:
            raise ValueError(
                "Use custom compute_metrics() function may cause extra high GPU memory footprint during evaluation, "
                "so eval_accumulation_steps should be set explicitly. If you are unsure of what size to set it to, it's"
                " recommended to set it to 1."
            )

        self.optimizer, self.lr_scheduler = optimizers
        # Check optimizer and lr_scheduler
        if self.optimizer is not None and not isinstance(self.optimizer, torch.optim.Optimizer):
            raise ValueError("`optimizer` must be the torch.optim.Optimizer type.")

        report_callbacks = [TensorBoardCallback]
        # Add additional tensorboard callback.
        if additional_tensorboard_hook is not None and len(additional_tensorboard_hook) > 0:
            report_callbacks.append(additional_tensorboard_hook[0])
        default_callbacks = DEFAULT_CALLBACKS + report_callbacks
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        if self.args.should_save:
            if not self._write_safely(os.makedirs, self.args.output_dir, exist_ok=True):
                raise OSError(f"Creating directory '{self.args.output_dir}' failed.")

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if (
            train_dataset is not None
            and isinstance(train_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            raise ValueError("the `--group_by_length` option is only available for `Dataset`, not `IterableDataset")

        self._signature_columns = None

        # Mixed precision setup
        self.do_grad_scaling = False
        if args.fp16 or args.bf16:
            if args.device == torch.device("cpu"):
                raise ValueError("CPU device is not supported when using auto mix percision.")
            self.amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
            #  bf16 does not need grad scaling
            self.do_grad_scaling = self.amp_dtype == torch.float16
            if self.do_grad_scaling:
                if self.atorch_fsdp:
                    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler as FSDPShardedGradScaler

                    self.scaler = FSDPShardedGradScaler()
                else:
                    self.scaler = torch.cuda.amp.GradScaler()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
        )

        if self.args.peft_type is not None and not isinstance(self.model, PeftModel):
            raise ValueError(
                f"'peft_type' is {self.args.peft_type}, while model type is not 'PeftModel' but {type(self.model)}."
            )
        elif isinstance(self.model, PeftModel) and self.args.peft_type is None:
            raise ValueError(f"Model type is {type(self.model)}, but 'peft_type' is not set.")

        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        if isinstance(self.model, PeftModel):
            base_model = self.model.get_base_model()
        else:
            base_model = self.model
        default_label_names = find_labels(base_model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.logit_names = ["logits"] if self.args.logit_names is None else self.args.logit_names
        self.model_forward_args = list(inspect.signature(base_model.forward).parameters.keys())
        self.can_return_loss = can_return_loss(base_model.__class__)

        # Set ATorch Parameters
        self.prepare_input = args.prepare_input
        self.optim_func = args.optim_func
        self.optim_args = args.optim_args
        self.optim_param_func = args.optim_param_func
        self.loss_func = args.loss_func
        self.distributed_sampler_cls = args.distributed_sampler_cls
        self.model_input_format = args.model_input_format
        self.load_strategy = load_strategy
        self.use_hpu_adamw = args.use_hpu_adamw

        if args.load_best_model_at_end and args.load_by_streaming:
            raise ValueError(
                "The method of loading FSDP model in ATorch is only allowed when calling "
                "auto_accelerate() function, not allowed at training end."
            )

        self._train_batch_size = self.args.train_batch_size

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing and args.atorch_checkpoint_cls is None:
            self.model.gradient_checkpointing_enable()

        # Resume from checkpoint
        # This time the call to _load_from_checkpoint() should be before _atorch_init()
        # Can't load fsdp+lora model from checkpoint after defining the model.
        if (
            args.resume_from_checkpoint is not None
            and os.path.exists(args.resume_from_checkpoint)
            and not (self.args.load_by_streaming or self.args.save_by_streaming)
        ):
            self._load_from_checkpoint(args.resume_from_checkpoint)

        # Call ATorch's auto_accelerate() to wrap model, optimizer, loss_function, ...
        self._atorch_init()

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformer.TrainerCallback`].

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """
        Remove a callback from the current list of [`~transformer.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformer.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def _atorch_init(self):
        if self.load_strategy is None:
            self.load_strategy = []
            # Set parallel mode
            if self.args.atorch_parallel_mode:
                self.load_strategy.append(("parallel_mode", ([("data", self.args.world_size)], None)))
            # Set module replace
            if (
                self.args.atorch_module_replace
                and not is_torch_npu_available()
                and not self.args.load_model_on_rank0_and_dispatch
            ):
                self.load_strategy.append("module_replace")
            # Set FSDP
            if self.atorch_fsdp:
                atorch_fsdp_config = {
                    "atorch_wrap_cls": self.args.atorch_wrap_cls,
                    "cpu_offload": self.args.cpu_offload,
                    "sync_module_states": self.args.sync_module_states,
                    "use_orig_params": self.args.use_orig_params,
                    "limit_all_gathers": self.args.limit_all_gathers,
                    "forward_prefetch": self.args.forward_prefetch,
                }
                if self.args.peft_type is None:
                    if self.args.load_model_on_rank0_and_dispatch:
                        atorch_fsdp_config["param_init_fn"] = lambda module: (
                            module.to_empty(device=torch.device("cuda"), recurse=False)
                            if self.args.process_index != 0
                            else None
                        )
                else:
                    if self.args.fsdp_flat_ckpt_path is not None and os.path.exists(self.args.fsdp_flat_ckpt_path):
                        ckpt_config = FSDPCkptConfig(
                            flat_ckpt_path=self.args.fsdp_flat_ckpt_path,
                            lora_ckpt_path=self.args.fsdp_lora_ckpt_path,
                            lora_prefix=self.args.fsdp_lora_prefix,
                            lora_cls=self.args.fsdp_lora_cls,
                            lora_weight_name=self.args.fsdp_lora_weight_name,
                        )
                        atorch_fsdp_config["param_init_fn"] = FSDPInitFn(
                            self.model, self.args.process_index, ckpt_config, self.args.atorch_wrap_cls
                        )
                    if self.args.wrap_trainable_outmost:
                        atorch_fsdp_config["wrap_trainable_outmost"] = True
                self.load_strategy.append(("fsdp", atorch_fsdp_config))

                # For gathering FSDP model parameters and saving them.
                self.save_policy = FullStateDictConfig(
                    offload_to_cpu=self.args.world_size > 1,
                    rank0_only=self.args.world_size > 1,
                )

            if self.args.bf16 or self.args.fp16:
                if self.args.bf16:
                    amp_config = {"dtype": torch.bfloat16, "skip_if_nonfinite": self.args.skip_if_nonfinite}
                else:
                    amp_config = {"dtype": torch.float16}
                self.load_strategy.append(("amp_native", amp_config))
            if self.args.gradient_checkpointing and self.args.atorch_checkpoint_cls is not None:
                self.load_strategy.append(("checkpoint", self.args.atorch_checkpoint_cls))

        # Get load_strategy with `Strategy` type by calling get_strategy function,
        # which can check the format of load_strategy
        status, self.load_strategy = get_strategy(self.load_strategy)
        if not status:
            raise TypeError("Unsupported load_strategy, please check your load_strategy.")

        # The listed methods will not change the model parameters, while other methods will.
        optim_methods_to_check = ["parallel_mode", "amp_native", "checkpoint"]

        if self.optimizer is not None:
            for opt_name, config, tunable in self.load_strategy:
                if opt_name not in optim_methods_to_check:
                    raise ValueError(
                        f"If you're using optimization methods outside of {optim_methods_to_check}, passing"
                        " `optimizers=(xxx,xxx)` when creating a trainer is not supported because auto_accelerate()"
                        " will change the model parameters. Please set `optimizers` via `args.optim_func` instead."
                    )

        # atorch auto_accelerate will restore batch_size by dividing by world size.
        train_dataloader_args = {
            "shuffle": self.args.shuffle,
            "batch_size": self._train_batch_size * self.args.world_size,
            "pin_memory": self.args.dataloader_pin_memory,
            "num_workers": self.args.dataloader_num_workers,
            "persistent_workers": self.args.dataloader_num_workers > 0,
        }

        if self.data_collator is not None:
            train_dataloader_args["collate_fn"] = self.data_collator

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            train_dataloader_args["drop_last"] = self.args.dataloader_drop_last

        if self.optim_args is None:
            self.optim_args = {
                "lr": self.args.learning_rate,
                "weight_decay": self.args.weight_decay,
                "eps": self.args.adam_epsilon,
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
            }

        find_unused_parameters = False
        if self.args.atorch_opt == "ddp":
            if self.args.ddp_find_unused_parameters is not None:
                find_unused_parameters = self.args.ddp_find_unused_parameters
            elif isinstance(self.model, PreTrainedModel):
                find_unused_parameters = not self.args.gradient_checkpointing
            elif self.args.peft_type not in ["lora", "qlora"]:
                find_unused_parameters = True

        if self.use_hpu_adamw:
            if is_torch_npu_available():
                from atorch.npu.optim import NpuAdamW

                logger.info("Use ATorch HPU Adamw to accelerate training.")
                self.optim_func = NpuAdamW
            else:
                logger.info("Not in HPU env, ignore 'use_hpu_adamw'")

        status, result, best_strategy = auto_accelerate(
            model=self.model,
            optim_func=self.optim_func,
            dataset=self.train_dataset if self.args.use_atorch_dataloader else None,
            distributed_sampler_cls=self.distributed_sampler_cls,
            dataloader_args=train_dataloader_args,
            loss_func=self.loss_func,
            prepare_input=self.prepare_input,
            model_input_format=self.model_input_format,
            optim_args=self.optim_args,
            optim_param_func=self.optim_param_func if self.optim_param_func is not None else None,
            excluded=self.args.excluded,
            included=self.args.included,
            load_strategy=self.load_strategy,
            finetune_strategy=self.args.finetune_strategy,
            save_strategy_to_file=self.args.save_strategy_to_file,
            ignore_dryrun_on_load_strategy=self.args.ignore_dryrun_on_load_strategy,
            find_unused_parameters=find_unused_parameters,
            sampler_seed=self.args.seed,
        )
        assert status, f"auto_accelerate failed. status: {status}, result: {result}, best_strategy: {best_strategy}"
        logger.info(f"Best strategy is: {best_strategy}")

        self.model = result.model
        self.optimizer = self.optimizer if self.optimizer is not None else result.optim
        self.loss_func = result.loss_func
        self.train_dataloader = result.dataloader
        self.prepare_input = result.prepare_input

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(["labels", "label", "label_ids"] + self.label_names + self.model_forward_args)
            )

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]  # type: ignore[attr-defined]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):  # type: ignore[attr-defined]
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(
                self.train_dataset, datasets.Dataset  # type: ignore[attr-defined]
            ):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):  # type: ignore[attr-defined]
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        elif data_collator is not None:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if data_collator is not None:
            dataloader_params["collate_fn"] = data_collator

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size <= 1:
            return SequentialSampler(eval_dataset)
        else:
            return None

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):  # type: ignore[attr-defined]
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        elif data_collator is not None:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if data_collator is not None:
            dataloader_params["collate_fn"] = data_collator

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(test_dataset, datasets.Dataset):  # type: ignore[attr-defined]
            test_dataset = self._remove_unused_columns(test_dataset, description="test")
        elif data_collator is not None:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="test")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if data_collator is not None:
            dataloader_params["collate_fn"] = data_collator

        if not isinstance(test_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(test_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        # We use the same batch_size as for eval.
        return self.accelerator.prepare(DataLoader(test_dataset, **dataloader_params))

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        # Do atorch create optimizer
        raise NotImplementedError("`create_optimizer` is not implemented.")

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        # Do atorch create scheduler
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                (
                    self.args.custom_lr_scheduler_type
                    if self.args.custom_lr_scheduler_type is not None
                    else self.args.lr_scheduler_type
                ),
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )
        return self.lr_scheduler

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        args = self.args

        self.is_in_train = True

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                logger.warning(
                    f"No valid checkpoint found in output directory {args.output_dir}. Will train from scratch."
                )

        return self._inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _inner_training_loop(self, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):

        # Data loader and number of training steps
        if self.args.use_atorch_dataloader and self.train_dataloader is not None:
            train_dataloader = self.train_dataloader
        else:
            train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        if package_version_bigger_than("transformers", "4.31.0"):
            self.state.logging_steps = args.logging_steps
            self.state.eval_steps = args.eval_steps
            self.state.save_steps = args.save_steps

        if self.optimizer is None:
            raise ValueError("Optimizer is None, please set optimizer via `auto_accelerate` function of ATorch.")

        if self.lr_scheduler is None:
            self.create_scheduler(num_training_steps=max_steps, optimizer=self.optimizer)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {count_model_params(self.model)[1]:,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self.model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        # Asynchronous saving model and optimizer.
        if self.args.async_save:
            self._init_async_save()

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)
            elif hasattr(epoch_iterator.sampler, "set_epoch"):
                epoch_iterator.sampler.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                loss = self.training_step(self.model, inputs)

                if args.logging_nan_inf_filter and (torch.isnan(loss) or torch.isinf(loss)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += loss

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or is_last_step_and_steps_less_than_grad_acc
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        if self.do_grad_scaling:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(self.model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            self.model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # Optimizer step
                    optimizer_was_run = True
                    if self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()
                        optimizer_was_run = not (
                            hasattr(self.optimizer, "step_was_skipped") and self.optimizer.step_was_skipped
                        )

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    self.model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, self.model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, self.model, trial, epoch, ignore_keys_for_eval, epoch_end=True)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if self.args.world_size > 1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self.log(metrics)

        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=self.args.output_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from
        # the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # wait for all IO subprocess to finish
        if args.async_save:
            self._join()
            dist.barrier()

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _issue_warnings_after_load(self, load_result):
        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model._keys_to_ignore_on_save
            ):
                self.model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}.")

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, epoch_end=False):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            if (
                epoch_end
                and self.args.save_at_specific_epoch is not None
                and (epoch + 1) in self.args.save_at_specific_epoch
            ):
                checkpoint_folder = checkpoint_folder + f"-epoch-{epoch+1}"
            output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
            if not self._save_checkpoint(model, output_dir, trial, metrics=metrics):
                if self.args.async_save:
                    # Delete checkpoint directory with saving error after sending deleting signal
                    self.pipe1.send(PipeMessageEntity(AsyncCheckpointSignal.DELETE_CKPT, ckpt_path=output_dir))
                elif os.path.exists(output_dir):
                    shutil.rmtree(output_dir, ignore_errors=True)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            if self.args.world_size > 1:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                try:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                except Exception as e:
                    logger.info(
                        f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
                        "\nThis won't yield the same results as if the training had not been interrupted."
                    )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """
        If model weight exists, load it. Not support streaming loading of ATorch.
        """
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)

        if not any(
            os.path.exists(f)
            for f in [
                weights_file,
                safe_weights_file,
                weights_index_file,
                safe_weights_index_file,
                adapter_weights_file,
                adapter_safe_weights_file,
            ]
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
            # We load the model state dict on the CPU to avoid an OOM error.
            if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
            else:
                state_dict = torch.load(weights_file, map_location="cpu")

            # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
            # which takes *args instead of **kwargs
            load_result = model.load_state_dict(state_dict, False)
            # release memory
            del state_dict
            self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif is_peft_available() and isinstance(model, PeftModel):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint):
                    model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding "
                        "saving folders. Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            logger.info("Loading sharded checkpoint...")
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=False, prefer_safe=self.args.save_safetensors
            )
            self._issue_warnings_after_load(load_result)

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model
        if (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True
            if is_peft_available() and isinstance(model, PeftModel):
                # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                    if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                        model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
                        # Load_adapter has no return value present, modify it when appropriate.
                        from torch.nn.modules.module import _IncompatibleKeys

                        load_result = _IncompatibleKeys([], [])
                    else:
                        logger.warning(
                            "The intermediate checkpoints of PEFT may not be saved correctly, "
                            f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding "
                            "saving folders. Check some examples here: https://github.com/huggingface/peft/issues/96"  # noqa: E501
                        )
                        has_been_loaded = False
                else:
                    logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                    has_been_loaded = False
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                    state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                else:
                    state_dict = torch.load(best_model_path, map_location="cpu")

                # If the model is on the GPU, it still works!
                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
            if has_been_loaded:
                self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)):
            load_result = load_sharded_checkpoint(model, self.state.best_model_checkpoint, strict=False)
            self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        optimizer_file = os.path.join(checkpoint, OPTIMIZER_NAME)
        streaming_ckpt_dir = os.path.join(checkpoint, STREAMING_CKPT_DIR)
        scheduler_file = os.path.join(checkpoint, SCHEDULER_NAME)
        atorch_lora_optimizer_file = os.path.join(checkpoint, ATORCH_LORA_OPTIMIZER_NAME)

        if not (
            (
                os.path.isfile(optimizer_file)
                or os.path.isdir(streaming_ckpt_dir)
                or os.path.isfile(atorch_lora_optimizer_file)
            )
            and os.path.isfile(scheduler_file)
        ):
            logger.warning(
                "Can't load optimizer and scheduler, "
                "please check whether optimizer and scheduler checkpoint files exist."
            )
            return

        map_location = self.args.device if self.args.world_size > 1 else "cpu"
        if self.atorch_fsdp and isinstance(self.model, FSDP):

            def _load_optimizer_on_rank0(optimizer_path):
                full_osd = None
                # In FSDP, we need to load the full optimizer state dict on rank 0 and then shard it
                if self.is_world_process_zero():
                    logger.info(f"Loading optimizer {optimizer_path}")
                    full_osd = torch.load(optimizer_path)
                # call scatter_full_optim_state_dict on all ranks
                sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, self.model)
                self.optimizer.load_state_dict(sharded_osd)

            if self.args.load_by_streaming or self.args.save_by_streaming:
                if self.args.peft_type is None:
                    if not os.path.isdir(streaming_ckpt_dir):
                        raise FileNotFoundError(f"Can't find {streaming_ckpt_dir} directory!")
                    logger.info(f"Loading optimizer from {streaming_ckpt_dir}.")
                    sm = ShardOptim(streaming_ckpt_dir)
                    reshard_optim_state = sm.reshard_optim_state_dict(self.model)
                    self.optimizer.load_state_dict(reshard_optim_state)
                elif self.args.peft_type == "lora":
                    _load_optimizer_on_rank0(atorch_lora_optimizer_file)
                else:
                    raise ValueError(f"Loading {self.args.peft_type} optimizer is not supported!")
            else:
                _load_optimizer_on_rank0(optimizer_file)
        else:
            logger.info(f"Loading optimizer {optimizer_file}")
            self.optimizer.load_state_dict(torch.load(optimizer_file, map_location=map_location))

        with warnings.catch_warnings(record=True) as caught_warnings:
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
        reissue_pt_warnings(caught_warnings)
        if self.do_grad_scaling and os.path.isfile(os.path.join(checkpoint, SCALER_NAME)):
            self.scaler.load_state_dict(torch.load(os.path.join(checkpoint, SCALER_NAME)))

    def _save_checkpoint(self, model, output_dir, trial, metrics=None) -> bool:
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        self.store_flos()

        if not self._write_safely(os.makedirs, output_dir, exist_ok=True):
            return False

        # Save lr scheduler
        if self.args.should_save:
            with warnings.catch_warnings(record=True) as caught_warnings:
                if not self._write_safely(
                    torch.save,
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                ):
                    return False
                logger.info(f"Scheduler saved in {os.path.join(output_dir, SCHEDULER_NAME)}")
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                if not self._write_safely(
                    torch.save,
                    self.scaler.state_dict(),
                    os.path.join(output_dir, SCALER_NAME),
                ):
                    return False
                logger.info(f"Scaler saved in {os.path.join(output_dir, SCALER_NAME)}")

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            if not self._write_safely(self.state.save_to_json, os.path.join(output_dir, TRAINER_STATE_NAME)):
                return False
            logger.info(f"Trainer state saved in {os.path.join(output_dir, TRAINER_STATE_NAME)}")

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.world_size > 1:  # TODO: please reconsider judge condition
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if self.args.world_size <= 1:
            rng_path = os.path.join(output_dir, "rng_state.pth")
        else:
            rng_path = os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth")
        if not self._write_safely(torch.save, rng_states, rng_path):
            return False

        # Save Hyperparameters
        if self.args.hyper_parameters is not None and self.args.should_save:
            hyperparameter_save_path = os.path.join(output_dir, HYPER_PARAMETER_NAME)

            def _save_hyperparameter(hyper_parameters: dict):
                with open(hyperparameter_save_path, "w", encoding="utf-8") as f:
                    json.dump(hyper_parameters, f, ensure_ascii=False, indent=2)

            if not self._write_safely(_save_hyperparameter, self.args.hyper_parameters):
                return False
            logger.info(f"Hyperparamter saved in {hyperparameter_save_path}")

        # Save model
        if not self.save_model(output_dir, _internal_call=True):
            return False

        # Save optimizer
        if self.args.save_optimizer:
            if not self.save_optimizer(output_dir):
                return False

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=False, output_dir=self.args.output_dir)

        return True

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> bool:
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        if not self._write_safely(os.makedirs, output_dir, exist_ok=True):
            return False

        if self.atorch_fsdp and isinstance(self.model, FSDP):
            if self.args.save_by_streaming:
                if self.args.peft_type is None:
                    streaming_ckpt_dir = os.path.join(output_dir, STREAMING_CKPT_DIR)
                    if not self._write_safely(os.makedirs, streaming_ckpt_dir, exist_ok=True):
                        return False
                    logger.info(f"Saving streaming model to {streaming_ckpt_dir}")
                    # TODO: whether to use multiprocess to save model
                    if not self._write_safely(save_fsdp_flat_param, self.model, streaming_ckpt_dir):
                        return False
                    logger.info(f"Streaming model saved in {streaming_ckpt_dir}")
                elif self.args.peft_type == "lora":
                    if not self._write_safely(
                        save_lora_param, self.model, output_dir, lora_weight_name=ATORCH_LORA_WEIGHT_NAME
                    ):
                        return False
                    model = unwrap_model(self.model)
                    if not self._save_peft_config(model, output_dir):
                        return False
                    logger.info(f"Streaming lora model saved in {output_dir}")

                    if self.args.save_base_model:
                        # save base model
                        assert hasattr(self, "save_policy")
                        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, self.save_policy):
                            state_dict = self.model.state_dict()
                        if self.args.should_save:
                            model = unwrap_model(self.model)
                            if _internal_call and self.args.async_save:
                                self._async_save(
                                    self._save_peft_base_model,
                                    model,
                                    output_dir,
                                    state_dict=state_dict,
                                    checkpoint_dir=output_dir,
                                )
                            else:
                                if not self._write_safely(self._save_peft_base_model, model, output_dir, state_dict):
                                    return False
                else:
                    raise ValueError(f"Saving {self.args.peft_type} model is not supported when training with FSDP.")
            else:
                assert hasattr(self, "save_policy")
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, self.save_policy):
                    state_dict = self.model.state_dict()
                if self.args.should_save:
                    if _internal_call and self.args.async_save:
                        self._async_save(
                            self._save,
                            output_dir,
                            state_dict=state_dict,
                            checkpoint_dir=output_dir,
                        )
                    else:
                        if not self._write_safely(self._save, output_dir, state_dict):
                            return False
        elif self.args.should_save:
            model = unwrap_model(self.model)
            state_dict = model.state_dict()
            if _internal_call and self.args.async_save:
                self._async_save(
                    self._save,
                    output_dir,
                    state_dict=state_dict,
                    checkpoint_dir=output_dir,
                )
            else:
                if not self._write_safely(self._save, output_dir, state_dict):
                    return False
        dist.barrier()
        return True

    def _save(self, output_dir: str, state_dict) -> bool:
        """
        Save model in rank zero. `state_dict` is acquired outside.

        Return:
            (bool): Whether the model is saved successfully.
        """

        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        if not self._write_safely(os.makedirs, output_dir, exist_ok=True):
            return False
        logger.info(f"Saving model checkpoint to {output_dir}")

        model = self.model

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(model, supported_classes):
            model = unwrap_model(model)
            if isinstance(model, supported_classes):
                if not self._write_safely(
                    model.save_pretrained,
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=self.args.save_safetensors,
                    max_shard_size=self.args.max_shard_size,
                ):
                    return False

            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    if not self._write_safely(
                        safetensors.torch.save_file,  # type: ignore[attr-defined]
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                    ):
                        return False
                else:
                    if not self._write_safely(torch.save, state_dict, os.path.join(output_dir, WEIGHTS_NAME)):
                        return False
        else:
            if not self._write_safely(
                model.save_pretrained,
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
                max_shard_size=self.args.max_shard_size,
            ):
                return False

        # Save base model if it is type of PeftModel
        if isinstance(model, PeftModel) and self.args.save_base_model:
            if not self._save_peft_base_model(model, output_dir, state_dict):
                return False

        # Save tokenizer
        if self.tokenizer is not None:
            if not self._write_safely(self.tokenizer.save_pretrained, output_dir):
                return False

        # Good practice: save your training arguments together with the trained model
        # TODO: check whether self.args contains local function
        if not self._write_safely(
            torch.save,
            self.args.to_dict(),
            os.path.join(output_dir, TRAINING_ARGS_NAME),
        ):
            return False

        return True

    def _save_peft_base_model(self, model: PeftModel, output_dir: str, state_dict) -> bool:
        """
        Save peft model in rank zero. `state_dict` is acquired outside.

        Return:
            (bool): Whether the model is saved successfully.
        """
        assert isinstance(model, PeftModel), f"Model type required PeftModel, but get {type(model)}!"
        base_model = model.get_base_model()
        # Filter the peft params ...
        param_keys = list(state_dict.keys())
        base_model_state_dict = {}
        for key in param_keys:
            value = state_dict[key]
            if LORA_KEY in key:
                continue
            if PEFT_PARAM_PREFIX in key:
                new_key = key.replace(PEFT_PARAM_PREFIX, "")
                base_model_state_dict[new_key] = value
            else:
                base_model_state_dict[key] = value
        logger.info(f"Saving base model checkpoint to {output_dir}")
        if not self._write_safely(
            base_model.save_pretrained,
            output_dir,
            state_dict=base_model_state_dict,
            safe_serialization=self.args.save_safetensors,
            max_shard_size=self.args.max_shard_size,
        ):
            return False
        return True

    def _save_peft_config(self, model: PeftModel, output_dir: str) -> bool:
        assert isinstance(model, PeftModel), f"Model type required PeftModel, but get {type(model)}!"
        # save the config and change the inference mode to `True`
        base_model = model.get_base_model()
        peft_config: PeftConfig = model.peft_config["default"]
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = (
                base_model.__dict__.get("name_or_path", None)
                if peft_config.is_prompt_learning
                else base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = model._get_base_model_class(
                is_prompt_tuning=peft_config.is_prompt_learning,
            )
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        if self.args.should_save:
            if not self._write_safely(peft_config.save_pretrained, output_dir, auto_mapping_dict=auto_mapping_dict):
                return False
        peft_config.inference_mode = inference_mode
        return True

    def save_optimizer(self, output_dir: str) -> bool:
        """
        Save the optimizer.
        """
        # full optimizer state dict
        full_osd = None
        if self.atorch_fsdp and isinstance(self.model, FSDP):
            if self.args.save_by_streaming:
                if self.args.peft_type is None:
                    streaming_ckpt_dir = os.path.join(output_dir, STREAMING_CKPT_DIR)
                    if not self._write_safely(os.makedirs, streaming_ckpt_dir, exist_ok=True):
                        return False
                    logger.info(f"Saving optimizer to {streaming_ckpt_dir}")
                    if not self._write_safely(
                        save_fsdp_optim_param,
                        self.model,
                        self.optimizer,
                        streaming_ckpt_dir,
                    ):
                        return False
                    logger.info(f"Optimizer saved in {streaming_ckpt_dir}")
                elif self.args.peft_type == "lora":
                    if not self._write_safely(
                        save_lora_optim_param,
                        self.model,
                        self.optimizer,
                        output_dir,
                        lora_weight_name=ATORCH_LORA_OPTIMIZER_NAME,
                    ):
                        return False
                    logger.info(f"Lora optimizer saved in {output_dir}")
                else:
                    raise ValueError(
                        f"Saving {self.args.peft_type} optimizer is not supported when training with FSDP."
                    )
            else:
                assert hasattr(self, "save_policy")
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, self.save_policy):
                    # may be removed after PyTorch 2.2
                    full_osd = FSDP.full_optim_state_dict(self.model, self.optimizer)
        else:
            full_osd = self.optimizer.state_dict()
        if self.args.should_save and full_osd is not None:
            optimizer_file = os.path.join(output_dir, OPTIMIZER_NAME)
            logger.info(f"Saving optimizer to {optimizer_file}")
            if self.args.async_save:
                self._async_save(
                    lambda state_dict, save_path: torch.save(state_dict, save_path),
                    state_dict=full_osd,
                    save_path=optimizer_file,
                    checkpoint_dir=output_dir,
                )
            else:
                if not self._write_safely(torch.save, full_osd, optimizer_file):
                    return False
                logger.info(f"Optimizer saved to {optimizer_file}")
        dist.barrier()
        return True

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        try:
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, output)
        except Exception:
            if self.args.ignore_write_errors:
                logger.exception("Logging failed! Maybe no space left when tensorboard writting!")
            else:
                raise

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        return self.args.process_index == 0

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (
            NameError,
            AttributeError,
            TypeError,
        ):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def autocast_smart_context_manager(self, cache_enabled: Optional[bool] = True):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        if self.args.fp16 or self.args.bf16:
            ctx_manager = torch.cuda.amp.autocast(cache_enabled=cache_enabled, dtype=self.amp_dtype)
        else:
            ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

        return ctx_manager

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if is_distributed():
            tensors = distributed_concat(tensors)
        return tensors

    def _get_learning_rate(self):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]["lr"]
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
        return last_lr

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        if self.args.world_size > 0:
            self.state.total_flos += (
                distributed_broadcast_scalars([self.current_flos], device=self.args.device).sum().item()
            )
            self.current_flos = 0
        else:
            self.state.total_flos += self.current_flos
            self.current_flos = 0

    def _sorted_checkpoints(
        self, output_dir=None, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False
    ) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [
            str(x)
            for x in Path(output_dir).glob(f"{checkpoint_prefix}-*")
            if os.path.isdir(x) and "epoch" not in str(x)
        ]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]  # type: ignore[misc]
        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))  # type: ignore[arg-type] # noqa: E501
            for i in range(best_model_index, len(checkpoints_sorted) - 2):
                checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]
        return checkpoints_sorted  # type: ignore[return-value]

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and checkpoints_sorted[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            if self.args.async_save:
                self.pipe1.send(PipeMessageEntity(AsyncCheckpointSignal.DELETE_CKPT, ckpt_path=checkpoint))
            else:
                shutil.rmtree(checkpoint, ignore_errors=True)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self.model

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and step % args.eval_accumulation_steps == 0
                and self.accelerator.sync_gradients
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # model.train() is necessary for methods such as Dropout.
        model.train()
        inputs = (
            self.prepare_input(inputs, self.args.device)
            if self.prepare_input is not None
            else self._prepare_inputs(inputs)
        )

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = (
            self.prepare_input(inputs, self.args.device)
            if self.prepare_input is not None
            else self._prepare_inputs(inputs)
        )
        if ignore_keys is None:
            unwrapped_model = unwrap_model(self.model)
            if hasattr(unwrapped_model, "config"):
                ignore_keys = getattr(unwrapped_model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.autocast_smart_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

            else:
                loss = None
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

            logits = None
            if isinstance(outputs, dict):
                logits = tuple(outputs.get(name) for name in self.logit_names)
            else:
                if self.args.logit_index >= 0:
                    logits = (outputs[self.args.logit_index],)

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})  # type: ignore[call-arg]
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.args.device)
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError("The batch received was empty, your model won't be able to train on it.")
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        return inputs

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.loss_func is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.loss_func(outputs, labels)
            if loss is None:
                raise ValueError("loss is None, please check your loss function")
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _secs2timedelta(secs):
        """
        convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimals
        """

        msec = int(abs(secs - int(secs)) * 100)
        return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"

    def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Reformat Trainer metrics values to a human-readable format

        Args:
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predict

        Returns:
            metrics (`Dict[str, float]`): The reformatted metrics
        """

        metrics_copy = {}
        for k, v in metrics.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{ int(v) >> 20 }MB"
            elif "_runtime" in k:
                metrics_copy[k] = self._secs2timedelta(v)
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            else:
                metrics_copy[k] = str(round(v, 4))

        return metrics_copy

    def log_metrics(self, split, metrics):
        if not self.is_world_process_zero():
            return

        logger.info(f"***** {split} metrics *****")
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            logger.info(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

    def save_metrics(self, split, metrics, combined=True):
        """
        Save metrics into a json file for that split, e.g. `train_results.json`.

        Under distributed environment this is done only for a process with rank 0.

        Args:
            split (`str`):
                Mode/split name: one of `train`, `eval`, `test`, `all`
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predict
            combined (`bool`, *optional*, defaults to `True`):
                Creates combined metrics by updating `all_results.json` with metrics of this call

        To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
        unformatted numbers are saved in the current method.

        """
        if not self.is_world_process_zero():
            return

        path = os.path.join(self.args.output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        if combined:
            path = os.path.join(self.args.output_dir, "all_results.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            all_metrics.update(metrics)
            with open(path, "w") as f:
                json.dump(all_metrics, f, indent=4, sort_keys=True)

    def save_state(self):
        """
        Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model

        Under distributed environment this is done only for a process with rank 0.
        """
        if not self.is_world_process_zero():
            return

        path = os.path.join(self.args.output_dir, "trainer_state.json")
        self.state.save_to_json(path)

    def _write_safely(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception:
            if self.args.ignore_write_errors:
                logger.exception(f"{func.__name__} writting error!")
                return False
            else:
                raise
        return True

    def _save_state_dict(self, checkpoint_dir, save_func, *args, **kwargs):
        save_success = self._write_safely(save_func, *args, **kwargs)

        if not save_success:
            self.pipe1.send(
                PipeMessageEntity(AsyncCheckpointSignal.DELETE_CKPT, pid=os.getpid(), ckpt_path=checkpoint_dir)
            )
        self.pipe1.send(PipeMessageEntity(AsyncCheckpointSignal.SAVE_OVER, pid=os.getpid(), ckpt_path=checkpoint_dir))
        logger.info(
            f"Asynchronous saving with subprocess pid {os.getpid()} in {checkpoint_dir} is over! "
            f"Save successfully? {'Yes' if save_success else 'No'}"
        )
        return save_success

    def _terminate_process_by_pid(self, pid: int):
        try:
            p = psutil.Process(pid)
            if p.is_running():
                p.terminate()
        except psutil.NoSuchProcess:
            logger.info(f"No process found with PID {pid}, maybe was terminated.")
        except psutil.TimeoutExpired:
            logger.info(f"Wait for process {pid} to finish timed out.")
            p.terminate()
        except Exception:
            if self.args.ignore_write_errors:
                logger.exception(f"Error occured when getting process handle by PID {pid}!")
            else:
                raise

    def _async_save(self, save_func, *args, **kwargs):
        if "checkpoint_dir" not in kwargs or "state_dict" not in kwargs:
            raise ValueError("Please pass 'checkpoint_dir' and 'state_dict' argument to _async_save() function.")

        # Get extra arguments from kwargs.
        checkpoint_dir = kwargs.pop("checkpoint_dir")

        # Wait if len(self.writing_processes) >= max_num_writing_processes
        # Two processes will be created to save model and optimizer respectively during every saving.
        start = time.time()
        while len(self.writing_processes) >= self.args.max_num_writing_processes:
            time.sleep(1)
            if time.time() - start > self.args.async_timeout:
                pids = list(self.writing_processes.keys())
                pids.sort()
                logger.info(f"Kill the earliest process {pids[0]}")
                # Kill the earliest process
                self._terminate_process_by_pid(pids[0])
                self.writing_processes.pop(pids[0])
                break

        # Move state dict from device to CPU
        state_dict = kwargs.pop("state_dict")
        state_dict = torch.utils._pytree.tree_map(lambda x: x.cpu() if isinstance(x, torch.Tensor) else x, state_dict)
        kwargs["state_dict"] = state_dict

        p = Process(target=self._save_state_dict, args=(checkpoint_dir, save_func, *args), kwargs=kwargs)
        p.start()
        self.writing_processes[p.pid] = checkpoint_dir

    def _async_manager(self):
        def _delete_ckpt(recv: PipeMessageEntity):
            checkpoint_dir = recv.ckpt_path
            logger.info(f"Deleting {checkpoint_dir}")

            # Stop the saving process with the working directory 'checkpoint_dir'
            process_to_del = []
            for pid, ckpt in self.writing_processes.items():
                if ckpt == checkpoint_dir:
                    self._terminate_process_by_pid(pid)
                    process_to_del.append(pid)
            for pid in process_to_del:
                self.writing_processes.pop(pid)

            # Delete 'checkpoint_dir'
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            logger.info(f"Checkpoint '{checkpoint_dir}' deleted!")

        def _save_ckpt_post(recv: PipeMessageEntity):
            pid = recv.pid
            logger.info(f"Destroying process {pid}")
            # Remove finished process from self.writing_processes
            self.writing_processes.pop(pid)
            self._terminate_process_by_pid(pid)

        train_is_over = False
        while True:
            # Receive data from the pipe with non-blocking way.
            if self.pipe2.poll():
                recv: PipeMessageEntity = self.pipe2.recv()
                if recv.signal_type == AsyncCheckpointSignal.DELETE_CKPT:
                    _delete_ckpt(recv)
                elif recv.signal_type == AsyncCheckpointSignal.SAVE_OVER:
                    _save_ckpt_post(recv)
                elif recv.signal_type == AsyncCheckpointSignal.TRAIN_OVER:
                    logger.info("Join all saving processes!")
                    train_is_over = True

                    # Begin to wait the remain IO operations to complete.
                    start = time.time()
                else:
                    raise ValueError(
                        f"Receive error signal type! Signal type should be one of "
                        f"{AsyncCheckpointSignal._member_names_}."
                    )
            else:
                time.sleep(1)

            if train_is_over:
                if (time.time() - start) > self.args.async_timeout:
                    logger.warning(
                        "The waiting time is too small to complete the remain asynchronous IO operations in "
                        f"{list(self.writing_processes.values())}."
                        "Please increase the waiting time by setting the argument 'async_timeout'."
                    )
                    # Terminate the remain subprocess forcibly.
                    for pid, ckpt in self.writing_processes.items():
                        self._terminate_process_by_pid(pid)
                    break
                if len(self.writing_processes) == 0:
                    logger.info("All asynchronous IO operations are finished!")
                    break

    def _init_async_save(self):
        if self.args.should_save:
            self.data_manager = Manager()

            # Duplex Pipe for inter-process communication
            self.pipe1, self.pipe2 = Pipe()

            # Record writing processes: [pid, checkpoint_dir]
            self.writing_processes: Dict[int, str] = self.data_manager.dict()

            self.manager_process = Process(target=self._async_manager)
            self.manager_process.start()

            # to disable TOKENIZERS_PARALLELISM=(true | false) warning
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _join(self):
        if self.args.should_save:
            self.pipe1.send(PipeMessageEntity(AsyncCheckpointSignal.TRAIN_OVER))
            # Wait for the manager_process to end
            self.manager_process.join(timeout=self.args.async_timeout)
            self.manager_process.terminate()

            # Shutdown the data manager
            self.data_manager.shutdown()
