import dataclasses
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import skip_first_batches
from accelerate.utils import is_deepspeed_available, set_seed
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers.data.data_collator import DataCollator, default_data_collator

# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import TensorBoardCallback

# isort: on
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import (
    CallbackHandler,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import IterableDatasetShard, distributed_concat
from transformers.trainer_utils import TrainOutput, has_length, speed_metrics

from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import is_distributed
from atorch.trainer.args import AtorchTrainingArgs
from atorch.trainer.atorch_profiler import get_profiler
from atorch.trainer.base.atorch_module import AtorchIRModel
from atorch.trainer.base.atorch_train_engine import AtorchTrainEngine
from atorch.trainer.base.dataset import AtorchDataset
from atorch.trainer.trainer_callback import FlowCallbackV2
from atorch.trainer.utils import DistributedType
from atorch.utils.hooks import ATorchHooks
from atorch.utils.import_util import is_megatron_lm_available, is_torch_npu_available
from atorch.utils.version import package_version_bigger_than

if is_megatron_lm_available():
    from megatron.training.global_vars import get_args, get_timers

    from atorch.trainer.megatron import AtorchMegatronEngine

if is_torch_npu_available():
    from torch_npu.profiler import dynamic_profile as dp


DEFAULT_FLOW_CALLBACKS = [FlowCallbackV2]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

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


def recursive_jsons_to_dict(value):
    if isinstance(value, str):
        try:
            parsed_value = json.loads(value)
            if isinstance(parsed_value, dict):
                return {k: recursive_jsons_to_dict(v) for k, v in parsed_value.items()}
            else:
                return parsed_value
        except json.JSONDecodeError:
            return value
    elif isinstance(value, dict):
        return {k: recursive_jsons_to_dict(v) for k, v in value.items()}
    else:
        return value


class AtorchTrainerV2:
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, AtorchIRModel] = None,
        args: AtorchTrainingArgs = None,
        data_collator: Optional[DataCollator] = None,
        datasets: Union[AtorchDataset, Tuple[Dataset, Optional[Dataset], Optional[Dataset]], None] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
    ):
        self.args: AtorchTrainingArgs = args
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.optimizer, self.lr_scheduler = optimizers
        self.data_collator = data_collator

        # Step up all device.
        self.args._setup_devices

        self.train_engine: Optional[AtorchTrainEngine] = None

        self.distributed_type = self.args.distributed_state.distributed_type

        if self.args.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Set seed
        # TODO: Consider when set_seed() should be called
        if self.args.seed is not None and self.distributed_type != DistributedType.MEGATRON:
            set_seed(self.args.seed)

        if self.distributed_type == DistributedType.MEGATRON and args.gradient_accumulation_steps > 1:
            raise NotImplementedError("Gradient accumulate is not supported when using Megatron.")

        # TODO: implement a TensorBoardCallback to be compatible with Megatron
        report_callbacks = []
        if self.distributed_type != DistributedType.MEGATRON:
            report_callbacks.append(TensorBoardCallback)
        # Add additional tensorboard callback.
        if additional_tensorboard_hook is not None and len(additional_tensorboard_hook) > 0:
            report_callbacks.append(additional_tensorboard_hook[0])
        default_callbacks = DEFAULT_FLOW_CALLBACKS + report_callbacks
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.distributed_type != DistributedType.MEGATRON:
            self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        self.state = TrainerState(
            is_local_process_zero=self.args.is_local_main_process,
            is_world_process_zero=self.args.is_main_process,
        )

        self.control = TrainerControl()

        # Ensure this is called at the end of __init__()
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def add_callback(self, callback: TrainerCallback):
        """
        Add a callback to the current list of [`~transformer.TrainerCallback`].

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback: TrainerCallback):
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

    @property
    def use_distributed(self):
        return self.distributed_type != DistributedType.NO_DISTRIBUTE

    def train(
        self,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        self.is_in_train = True

        resume_from_checkpoint_arg_in_kwargs = kwargs.get("resume_from_checkpoint", None)
        if resume_from_checkpoint_arg_in_kwargs is not None:
            logger.warning(
                "'resume_from_checkpoint' pass by train function is deprecated and will have NO effect, "
                "please set it in training args."
            )

        resume_from_checkpoint = self.args.resume_from_checkpoint
        extra_dict = {}

        return self._inner_training_loop(
            args=self.args, resume_from_checkpoint=resume_from_checkpoint, ais_saved_extra_dict=extra_dict
        )

    def _inner_training_loop(self, args: AtorchTrainingArgs, resume_from_checkpoint=None, **kwargs):
        if self.datasets is not None:
            dataset_num = len(self.datasets)

            train_dataset = self.datasets[0] if dataset_num > 0 else None
            eval_dataset = self.datasets[1] if dataset_num > 1 else None
            test_dataset = self.datasets[2] if dataset_num > 2 else None

            # Log a few random samples from the training set:
            # for index in random.sample(range(len(train_dataset)), 3):
            #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

            train_dataloader = None
            eval_dataloader = None
            test_dataloader = None

            # DataLoaders creation:
            if train_dataset is not None:
                train_dataloader = DataLoader(
                    train_dataset,
                    shuffle=True,
                    collate_fn=default_data_collator if self.data_collator is None else self.data_collator,
                    batch_size=args.per_device_train_batch_size,
                )
            if eval_dataset is not None:
                eval_dataloader = DataLoader(
                    eval_dataset,
                    collate_fn=default_data_collator if self.data_collator is None else self.data_collator,
                    batch_size=args.per_device_eval_batch_size,
                )
            if test_dataset is not None:
                test_dataloader = DataLoader(
                    test_dataset,
                    collate_fn=default_data_collator if self.data_collator is None else self.data_collator,
                    batch_size=args.per_device_eval_batch_size,
                )

            dataloaders = (train_dataloader, eval_dataloader, test_dataloader)
        else:
            dataloaders = None

        # Define an engine
        # TODO:
        # 1. TO implement DDP, FSDP, DeepSpeed engine
        if self.distributed_type == DistributedType.MULTI_GPU:
            raise NotImplementedError("Not implement DDP")
        elif self.distributed_type == DistributedType.FSDP:
            raise NotImplementedError("Not implement FSDP")
        elif self.distributed_type == DistributedType.DEEPSPEED:
            if is_deepspeed_available():
                raise ValueError("DeepSpeed is not installed => run `pip install deepspeed` or build it from source.")
            raise NotImplementedError("Not implement deepspeed")
        elif self.distributed_type == DistributedType.MEGATRON:
            if not is_megatron_lm_available():
                raise ValueError("Megatron-LM is not installed.")
            # TODO: check type
            self.train_engine = AtorchMegatronEngine(
                # TODO YY: adjust input
                train_args=args,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                dataloaders=dataloaders,
                resume_from_checkpoint=resume_from_checkpoint,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Not implemented distributed backend {self.distributed_type}.")

        train_dataloader = self.train_engine.get_dataloader("train")

        if args.is_local_main_process:
            logger.info("------------------------ AtorchTrainingArgs ------------------------")
            args_dict = args.to_dict()
            str_list = []
            for k, v in args_dict.items():
                dots = "-" * (48 - len(k))
                str_list.append("  {} {} {}".format(k, dots, v))
            for arg in sorted(str_list, key=lambda x: x.lower()):
                logger.info(arg)
            logger.info("-------------------- end of AtorchTrainingArgs ---------------------")

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        # TODO(1009): To compact micro_batch_size in Megatron.
        total_train_batch_size = args.global_train_batch_size * args.gradient_accumulation_steps

        if self.distributed_type == DistributedType.MEGATRON:
            megatron_args = get_args()
            if args.max_steps > 0 and args.max_steps != megatron_args.train_iters:
                logger.warning(
                    "args.max_steps will be overwritten by megatron_args.train_iters under MEGATRON training mode."
                )
            args.max_steps = megatron_args.train_iters

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
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs  # type: ignore[assignment] # noqa: E501
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

        # Not support search hyper param
        self.state.is_hyper_param_search = False
        if package_version_bigger_than("transformers", "4.31.0"):
            self.state.logging_steps = args.logging_steps
            self.state.eval_steps = args.eval_steps
            self.state.save_steps = args.save_steps

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        if self.distributed_type != DistributedType.MEGATRON:
            logger.info(f"  Number of trainable parameters = {count_model_params(self.model)[1]:,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None  # noqa: F841

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None:
            ais_saved_extra_dict = kwargs.get("ais_saved_extra_dict", None)

            if ais_saved_extra_dict:
                trainer_state_dict = ais_saved_extra_dict.get("trainer_state", None)
                if trainer_state_dict is None:
                    raise ValueError("not able to load trainer state from ais, please check with ais supporter")
                self.state = TrainerState(**trainer_state_dict)
                custom_dict = ais_saved_extra_dict.get("customize_dict", None)
                if args.distributed_type == "megatron":
                    print(f"custom_dict is {custom_dict}")
                    get_args().customize_dict = custom_dict
            else:
                if args.distributed_type == "megatron":
                    trainer_state_path = os.path.join(
                        resume_from_checkpoint, "iter_{:07d}".format(self.train_engine.iteration)
                    )
                else:
                    trainer_state_path = resume_from_checkpoint
                self.state = TrainerState.load_from_json(os.path.join(trainer_state_path, TRAINER_STATE_NAME))
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
        self.callback_handler.model = self.model  # TODO: Update to engine
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.args.is_local_main_process
        self.state.is_world_process_zero = self.args.is_main_process

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        self.train_engine.optimizer_zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Empty redundant memory.
        torch.cuda.empty_cache()

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break

        if self.distributed_type == DistributedType.MEGATRON:
            timers = get_timers()
            timers("interval-time", log_level=0).start(barrier=True)

        dist.barrier()
        with get_profiler(self.args) as prof:
            total_batched_samples = 0
            for epoch in range(epochs_trained, num_train_epochs):
                epoch_iterator = train_dataloader
                # TODO: remove the following code, and add set_epoch() method for AtorchDataloader()
                # if hasattr(epoch_iterator, "set_epoch"):
                #     epoch_iterator.set_epoch(epoch)
                # elif hasattr(epoch_iterator.sampler, "set_epoch"):
                #     epoch_iterator.sampler.set_epoch(epoch)

                # TODO: support past_index
                # # Reset the past mems state at the beginning of each epoch if necessary.
                # if args.past_index >= 0:
                #     self._past = None

                steps_in_epoch = (
                    len(epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                if (
                    epoch == epochs_trained
                    and resume_from_checkpoint is not None
                    and steps_trained_in_current_epoch == 0
                ):
                    self._load_rng_state(resume_from_checkpoint)

                rng_to_sync = False
                steps_skipped = 0
                if self.distributed_type != DistributedType.MEGATRON:
                    if steps_trained_in_current_epoch > 0:
                        epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                        steps_skipped = steps_trained_in_current_epoch
                        steps_trained_in_current_epoch = 0
                        rng_to_sync = True
                else:
                    if steps_trained_in_current_epoch > 0:
                        # TODO: steps_skipped should be get_args().iteration, because
                        # self.train_engine.iteration will be real-time updated.
                        steps_skipped = self.train_engine.iteration

                step = -1
                for step, inputs in enumerate(epoch_iterator):
                    self.train_engine.train()
                    total_batched_samples += 1
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # TODO: remove them.
                    # # Skip past any already trained steps if resuming training
                    # if steps_trained_in_current_epoch > 0:
                    #     steps_trained_in_current_epoch -= 1
                    #     if steps_trained_progress_bar is not None:
                    #         steps_trained_progress_bar.update(1)
                    #     if steps_trained_in_current_epoch == 0:
                    #         self._load_rng_state(resume_from_checkpoint)
                    #     continue
                    # elif steps_trained_progress_bar is not None:
                    #     steps_trained_progress_bar.close()
                    #     steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    if self.distributed_type == DistributedType.MEGATRON:
                        # MegatronEngine's train_step() contains:
                        # forward, backward, optimizer.step(), zero_grad()
                        loss = self.train_engine(inputs)
                    else:
                        with self.train_engine.accumulate():
                            loss = self.train_engine(**inputs)

                            self.train_engine.backward(loss)

                            self.train_engine.optimizer_step()
                            self.train_engine.scheduler_step()
                            self.train_engine.optimizer_zero_grad()

                    if args.logging_nan_inf_filter and (torch.isnan(loss) or torch.isinf(loss)):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += loss

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or is_last_step_and_steps_less_than_grad_acc
                    ):
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, self.model, epoch)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break

                    if self.args.profiler_type == "hw_dp":
                        dp.step()
                    elif prof is not None and not isinstance(prof, nullcontext):
                        prof.step()

                if step < 0:
                    logger.warning(
                        "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, self.model, epoch, epoch_end=True)

                if self.control.should_training_stop:
                    break

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        # self.store_flos()
        # metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(global_step=self.state.global_step, training_loss=train_loss, metrics=metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, epoch, epoch_end=False):
        if self.distributed_type == DistributedType.MEGATRON:
            timers = get_timers()

        if self.control.should_log:
            if self.distributed_type == DistributedType.MEGATRON:
                logging_metrics = self.train_engine.training_log()
                self.log(logging_metrics)
            else:
                logs: Dict[str, float] = {}

                # all_gather + mean() to get average loss over all processes
                tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

                logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
                logs["learning_rate"] = self._get_learning_rate()

                self._total_loss_scalar += tr_loss_scalar
                self._globalstep_last_logged = self.state.global_step

                self.log(logs)

            # reset tr_loss to zero
            tr_loss -= tr_loss

        metrics = None
        if self.control.should_evaluate:
            if self.distributed_type == DistributedType.MEGATRON:
                timers("interval-time").stop()
                metrics = self.evaluate()
                timers("interval-time", log_level=0).start(barrier=True)
            else:
                # TODO: Implement evaluate
                metrics = self.evaluate()

                # Run delayed LR scheduler now that metrics are populated
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric_to_check = self.args.metric_for_best_model
                    if not metric_to_check.startswith("eval_"):
                        metric_to_check = f"eval_{metric_to_check}"
                    self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            # Save model checkpoint
            if self.distributed_type == DistributedType.MEGATRON:
                timers("interval-time").stop()

                checkpoint_dir_path = self.train_engine.get_checkpoint_path_dir(
                    self.args.output_dir, return_base_dir=True
                )

                if self.args.is_main_process:
                    os.makedirs(checkpoint_dir_path, exist_ok=True)

                torch.distributed.barrier()

                self.train_engine.save_checkpoint(
                    self.args.output_dir,
                    best_model_checkpoint=self.state.best_model_checkpoint,
                    trainer_state=dataclasses.asdict(self.state),
                )

                if self.args.is_main_process:
                    checkpoint_dir_path = self.train_engine.get_checkpoint_path_dir(
                        get_args().save, return_base_dir=True
                    )
                    self.state.save_to_json(os.path.join(checkpoint_dir_path, TRAINER_STATE_NAME))

                timers("interval-time", log_level=0).start(barrier=True)
            else:
                # TODO: Save checkpoint for other cases
                raise NotImplementedError("Not implemented saving checkpoint.")

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.distributed_type != DistributedType.MEGATRON:
            if self.state.epoch is not None:
                logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        # self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, output)

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
        if self.distributed_type == DistributedType.MEGATRON:
            eval_dataloader = self.train_engine.get_dataloader("eval")
            megatron_args = get_args()
            if megatron_args.eval_iters > 0:
                max_eval_steps = megatron_args.eval_iters
            else:
                max_eval_steps = len(eval_dataloader)
        else:
            # max_eval_steps = self.num_examples(eval_dataloader)
            raise ValueError(f"Evaluation on {self.distributed_type} not implement.")

        self.train_engine.eval()

        batch_size = self.args.global_eval_batch_size

        logger.info("***** Running Evaluation *****")
        if has_length(eval_dataloader):
            logger.info(f"  Num examples = {self.num_examples(eval_dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                if self.distributed_type == DistributedType.MEGATRON:
                    outputs = self.train_engine(batch)
                else:
                    outputs = self.train_engine(**batch)

            if self.train_engine.train_step_handler.model_output_class is not None:
                loss = outputs.loss
            else:
                loss = outputs
            # New Code
            # For Megatron-LM, the losses are already averaged across the data parallel group
            if self.distributed_type == DistributedType.MEGATRON:
                losses.append(loss)
            else:
                # TODO: Implement loss gathering.
                pass

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            if step >= max_eval_steps - 1:
                break
        try:
            if self.distributed_type == DistributedType.MEGATRON:
                losses = torch.tensor(losses)
            else:
                losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        eval_log = {"eval_loss": eval_loss, "perplexity": perplexity}

        self.log(eval_log)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_log)

        return eval_log

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
