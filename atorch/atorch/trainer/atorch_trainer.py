"""
We implement the trainer to be compatible with Transformers Trainer, and many codes are taken from Transformers Trainer
with some modifications.
"""

import contextlib
import datetime
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import safetensors
import torch
import torch.distributed as dist
from accelerate import Accelerator, skip_first_batches
from packaging import version
from peft.peft_model import PeftModel
from torch import nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import __version__, get_scheduler
from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator

import atorch
from atorch.auto import auto_accelerate
from atorch.distributed.distributed import is_distributed
from atorch.trainer.atorch_args import AtorchArguments
from atorch.utils.fsdp_save_util import ShardOptim, save_fsdp_flat_param, save_fsdp_optim_param
from atorch.utils.version import torch_version

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import get_reporting_integration_callbacks

# isort: on
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import OPTIMIZER_NAME, SCALER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME, TRAINING_ARGS_NAME
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
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
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalPrediction,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainOutput,
    find_executable_batch_size,
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
    find_labels,
    is_datasets_available,
    is_peft_available,
)

if TYPE_CHECKING:
    import optuna

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

PEFT_PARAM_PREFIX = "base_model.model."
LORA_KEY = "lora"
STREAMING_CKPT_DIR = "streaming_ckpt"  # save/load dir for Atorch's FSDP

logger = logging.getLogger(__name__)


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
        load_strategy: Optional[List[Union[str, Tuple]]] = None,
        **kwargs,
    ):
        if args is None:
            output_dir = "tmp_atorch_trainer"
            logger.info(f"No `AtorchArguments` passed, using `output_dir={output_dir}`.")
            args = AtorchArguments(output_dir=output_dir)  # type: ignore[call-arg]

        if len(args.sharded_ddp) > 0 or len(args.fsdp) > 0 or args.deepspeed is not None:
            logger.warning(
                "--sharded_ddp, --fsdp, --deepspeed in `TrainingArguments` is invalid when using `AtorchArguments`."
            )

        self.args = args
        self.kwargs = kwargs
        self.model = model

        self.atorch_fsdp = args.atorch_opt == "fsdp"

        if self.atorch_fsdp and torch_version() < (2, 0, 0):
            raise ValueError("Greater than version 2.0 of PyTorch is necessary for FSDP.")

        if args.save_load_by_streaming and not self.atorch_fsdp:
            raise ValueError("--atorch_opt fsdp is needed when using --save_load_by_streaming.")

        # create accelerator object
        self.accelerator = Accelerator()

        # force device and distributed setup init explicitly
        args._setup_devices

        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

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

        self.control = TrainerControl()
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.model_forward_args = list(inspect.signature(self.model.forward).parameters.keys())
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Set ATorch Parameters
        self.atorch_wrap_cls = args.atorch_wrap_cls
        self.prepare_input = args.prepare_input
        self.optim_param_func = args.optim_param_func
        self.loss_func = args.loss_func
        self.distributed_sampler_cls = args.distributed_sampler_cls
        self.model_input_format = args.model_input_format
        self.load_strategy = load_strategy

        self.is_peft_model = isinstance(model, PeftModel)

        if self.is_peft_model and args.save_load_by_streaming:
            raise ValueError("Atorch's streaming save/load does not support Peft finetune.")

        if args.load_best_model_at_end and args.save_load_by_streaming:
            raise ValueError(
                "Atorch's streaming save/load can only load model when defining model, "
                "not support to load model at end."
            )

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
        if self.load_strategy is not None:
            # TODO: check strategy format
            pass
        else:
            self.load_strategy = []
            # Set parallel mode
            if self.args.atorch_parallel_mode:
                self.load_strategy.append(("parallel_mode", ([("data", torch.distributed.get_world_size())], None)))
            # Set module replace
            if self.args.atorch_module_replace:
                self.load_strategy.append("module_replace")
            # Set FSDP
            if self.atorch_fsdp:
                atorch_fsdp_config = {
                    "atorch_wrap_cls": self.atorch_wrap_cls,
                    "cpu_offload": self.args.cpu_offload,
                    "sync_module_states": self.args.sync_module_states,
                    "use_orig_params": self.args.use_orig_params,
                    "limit_all_gathers": self.args.limit_all_gathers,
                    "forward_prefetch": self.args.forward_prefetch,
                }
                if self.is_peft_model and self.args.wrap_trainable_outmost:
                    atorch_fsdp_config["wrap_trainable_outmost"] = True
                self.load_strategy.append(("fsdp", atorch_fsdp_config))

            if self.args.bf16 or self.args.fp16:
                if self.args.bf16:
                    amp_config = {"dtype": torch.bfloat16}
                    self.load_strategy.append(("amp_native", amp_config))
                elif self.args.fp16:
                    amp_config = {"dtype": torch.float16}
                    self.load_strategy.append(("amp_native", amp_config))
            if self.args.gradient_checkpointing:
                self.load_strategy.append(("checkpoint", self.atorch_wrap_cls))

        # atorch auto_accelerate will restore batch_size by dividing by world size.
        train_dataloader_args = {
            "shuffle": True,
            "collate_fn": self.data_collator,
            "batch_size": self._train_batch_size * self.args.world_size,
            "pin_memory": self.args.dataloader_pin_memory,
            "num_workers": self.args.dataloader_num_workers,
            "persistent_workers": self.args.dataloader_num_workers > 0,
        }

        if not isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            train_dataloader_args["drop_last"] = self.args.dataloader_drop_last

        optim_args = {
            "lr": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "eps": self.args.adam_epsilon,
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
        }

        status, result, best_strategy = auto_accelerate(
            model=self.model,
            optim_func=self.optimizer,
            dataset=self.train_dataset if self.args.use_atorch_dataloader else None,
            distributed_sampler_cls=self.distributed_sampler_cls,
            dataloader_args=train_dataloader_args,
            loss_func=self.loss_func,
            prepare_input=self.prepare_input,
            model_input_format=self.model_input_format,
            optim_args=optim_args,
            optim_param_func=partial(self.optim_param_func, args=optim_args)
            if self.optim_param_func is not None
            else None,
            load_strategy=self.load_strategy,
            ignore_dryrun_on_load_strategy=self.args.ignore_dryrun_on_load_strategy,
        )
        assert status, f"auto_accelerate failed. status: {status}, result: {result}, best_strategy: {best_strategy}"
        logger.info(f"Best strategy is: {best_strategy}")

        self.model = result.model
        self.optimizer = result.optim
        self.lr_scheduler = result.lr_scheduler
        self.loss_func = result.loss_func
        self.train_dataloader = result.dataloader
        self.prepare_input = result.prepare_input

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names + self.model_forward_args))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
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

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
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
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
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
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

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

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

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
        # Do Atorch get test dataloader
        raise NotImplementedError("`get_test_dataloader` is not implemented.")

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
        raise NotImplementedError("`create_scheduler` is not implemented.")

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
        # This might change the seed so needs to run first.
        self._train_batch_size = self.args.train_batch_size

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                logger.warning(
                    f"No valid checkpoint found in output directory {args.output_dir}. Will train from scratch."
                )

        if resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint)

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        return inner_training_loop(
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        # Set model, optimizer, train dataloader
        self._atorch_init()

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

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.optimizer is None:
            raise ValueError("Optimizer is None, please set optimizer via `auto_accelerate` function of ATorch.")

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(max_steps),
                num_training_steps=max_steps,
            )

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

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

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
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or  # noqa: W504
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
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
            self._maybe_log_save_evaluate(tr_loss, self.model, trial, epoch, ignore_keys_for_eval)

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

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
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
            self._save_checkpoint(model, trial, metrics=metrics)
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
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        streaming_ckpt_dir = os.path.join(resume_from_checkpoint, STREAMING_CKPT_DIR)

        if not any(
            os.path.exists(f)
            for f in [
                weights_file,
                safe_weights_file,
                weights_index_file,
                safe_weights_index_file,
                adapter_weights_file,
                adapter_safe_weights_file,
                streaming_ckpt_dir,
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

        if os.path.exists(streaming_ckpt_dir):
            if self.args.save_load_by_streaming:
                logger.info(
                    "Attention: when using ATorch's streaming save/laod, "
                    "you should load checkpoint when defining your model."
                )
        elif os.path.isfile(weights_file) or os.path.isfile(safe_weights_file):
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
        best_streaming_ckpt_dir = os.path.join(self.state.best_model_checkpoint, STREAMING_CKPT_DIR)

        model = self.model
        if (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
            or os.path.exists(best_streaming_ckpt_dir)
        ):
            has_been_loaded = True
            if self.args.save_load_by_streaming:
                # TODO: can't load best model when using --save_load_by_streaming
                logger.warning("Can't load best model when using --save_load_by_streaming.")
            else:
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

        if not (
            (os.path.isfile(optimizer_file) or os.path.isdir(streaming_ckpt_dir)) and os.path.isfile(scheduler_file)
        ):
            logger.warning(
                "Can't load optimizer and scheduler, "
                "please check whether optimizer and scheduler checkpoint files exist."
            )
            return

        map_location = self.args.device if self.args.world_size > 1 else "cpu"
        if self.atorch_fsdp:
            if self.args.save_load_by_streaming:
                if not os.path.isdir(streaming_ckpt_dir):
                    raise FileNotFoundError(f"Can't find {streaming_ckpt_dir} directory!")
                logger.info("Begin to load optimizer by streaming")
                sm = ShardOptim(streaming_ckpt_dir)
                reshard_optim_state = sm.reshard_optim_state_dict(self.model)
                self.optimizer.load_state_dict(reshard_optim_state)
            else:
                logger.info("Load optimizer without streaming")
                full_osd = None
                # In FSDP, we need to load the full optimizer state dict on rank 0 and then shard it
                if self.is_world_process_zero():
                    full_osd = torch.load(optimizer_file)
                # call scatter_full_optim_state_dict on all ranks
                sharded_osd = self.model.__class__.scatter_full_optim_state_dict(full_osd, self.model)
                self.optimizer.load_state_dict(sharded_osd)
        else:
            self.optimizer.load_state_dict(torch.load(optimizer_file, map_location=map_location))

        with warnings.catch_warnings(record=True) as caught_warnings:
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
        reissue_pt_warnings(caught_warnings)
        if self.do_grad_scaling and os.path.isfile(os.path.join(checkpoint, SCALER_NAME)):
            self.scaler.load_state_dict(torch.load(os.path.join(checkpoint, SCALER_NAME)))

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        self.store_flos()

        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        # Save optimizer and scheduler
        if self.atorch_fsdp:
            if not isinstance(self.model, FSDP):
                raise ValueError("Self.model is not 'FullyShardedDataParallel' type when using --atorch_opt 'fsdp'.")
            if self.args.save_load_by_streaming:
                streaming_ckpt_dir = os.path.join(output_dir, STREAMING_CKPT_DIR)
                os.makedirs(streaming_ckpt_dir, exist_ok=True)
                logger.info(f"Saving optimizer to {streaming_ckpt_dir}")
                save_fsdp_optim_param(self.model, self.optimizer, streaming_ckpt_dir)
                logger.info(f"Optimizer saved to {streaming_ckpt_dir}")
            else:
                save_policy = FullStateDictConfig(
                    offload_to_cpu=self.args.cpu_offload, rank0_only=atorch.world_size() > 1
                )
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    # may be removed after PyTorch 2.2
                    full_osd = FSDP.full_optim_state_dict(self.model, self.optimizer)
        else:
            full_osd = self.optimizer.state_dict()
        if self.args.should_save:
            if not self.args.save_load_by_streaming:
                optimizer_file = os.path.join(output_dir, OPTIMIZER_NAME)
                logger.info(f"Saving optimizer to {optimizer_file}")
                torch.save(full_osd, optimizer_file)
                logger.info(f"Optimizer saved to {optimizer_file}")

            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

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
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

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

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=self.args.output_dir)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.atorch_fsdp:
            if not isinstance(self.model, FSDP):
                raise ValueError("Self.model is not 'FullyShardedDataParallel' type when using --atorch_opt 'fsdp'.")
            if self.args.save_load_by_streaming:
                if isinstance(unwrap_model(self.model), PeftModel):
                    raise ValueError(
                        "Non-PeftModel is required when using Atorch's streaming save function `save_fsdp_flat_param`."
                    )
                streaming_ckpt_dir = os.path.join(output_dir, STREAMING_CKPT_DIR)
                os.makedirs(streaming_ckpt_dir, exist_ok=True)
                logger.info(f"Saving streaming model to {streaming_ckpt_dir}")
                save_fsdp_flat_param(self.model, streaming_ckpt_dir)
                logger.info(f"Streaming model saved to {streaming_ckpt_dir}")
            else:
                save_policy = FullStateDictConfig(
                    offload_to_cpu=self.args.world_size > 1, rank0_only=self.args.world_size > 1
                )
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                    model_state_dict = self.model.state_dict()
                self._save(output_dir, state_dict=model_state_dict, save_base_model=self.args.save_base_model)
        else:
            self._save(output_dir, save_base_model=self.args.save_base_model)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _save(self, output_dir: Optional[str] = None, state_dict=None, save_base_model: bool = False):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            model = unwrap_model(self.model)
            if state_dict is None:
                state_dict = model.state_dict()
            if isinstance(model, supported_classes):
                if self.args.should_save:
                    model.save_pretrained(
                        output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                    )
                if isinstance(model, PeftModel) and save_base_model:
                    base_model = model.get_base_model()
                    state_dict = base_model.state_dict()
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
                    if self.args.should_save:
                        logger.info(f"Saving base model checkpoint to {output_dir}")
                        base_model.save_pretrained(
                            output_dir, state_dict=base_model_state_dict, safe_serialization=self.args.save_safetensors
                        )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.should_save:
                    if self.args.save_safetensors:
                        safetensors.torch.save_file(state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME))
                    else:
                        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            if self.args.should_save:
                self.model.save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )

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
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

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
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
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

        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

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
        self.eval_dataloader = self.get_eval_dataloader(eval_dataset)
        logger.info("Start evaluation")
        self.model.eval()
        losses = []
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                batch = {k: v.to(self.args.device) for k, v in batch.items()}
                with self.autocast_smart_context_manager():
                    loss = self.compute_loss(self.model, batch)
                repeated_loss = loss.repeat(self.args.per_device_eval_batch_size)
                if repeated_loss.ndim == 0:
                    repeated_loss = repeated_loss[None]
                output_tensors = [repeated_loss for _ in range(self.args.world_size)]
                torch.distributed.all_gather(output_tensors, repeated_loss)
                losses.append(torch.cat([t.detach().cpu() for t in output_tensors], dim=0))

        losses = torch.cat(losses)
        losses = losses[: len(self.eval_dataset)]
        mean_loss = torch.mean(losses).item()
        metrics = {"eval_loss": mean_loss, "step": self.state.global_step}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics

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
        # Do Atorch predict
        raise NotImplementedError("`predict` is not implemented.")

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
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach() / self.args.gradient_accumulation_steps

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

        print(f"***** {split} metrics *****")
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

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
