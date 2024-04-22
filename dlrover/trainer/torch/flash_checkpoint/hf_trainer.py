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

import os
import random
import re
import shutil
import warnings
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers import Trainer
from transformers.trainer import (
    OPTIMIZER_NAME,
    PREFIX_CHECKPOINT_DIR,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
    DeepSpeedSchedulerWrapper,
    ParallelMode,
    PeftModel,
    PreTrainedModel,
    is_peft_available,
    logger,
    reissue_pt_warnings,
    unwrap_model,
)

from dlrover.python.common.storage import PosixDiskStorage
from dlrover.trainer.torch.flash_checkpoint.deepspeed import (
    AsyncCheckpointAgent,
)
from dlrover.trainer.torch.flash_checkpoint.deepspeed_engine import (
    DeepSpeedCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.engine import CheckpointEngine
from dlrover.trainer.torch.flash_checkpoint.full_ckpt_engine import (
    FullCheckpointEngine,
)

torch_native_save = torch.save
torch_native_load = torch.load


class HfFlashCheckpointer(object):
    def __init__(self, checkpoint_dir, storage=None):
        self.checkpoint_dir = checkpoint_dir
        self.storage = PosixDiskStorage() if not storage else storage
        self.ckpt_agent = AsyncCheckpointAgent(self.storage)
        self.async_save_engine: Optional[CheckpointEngine] = None

    def save_checkpoint_to_memory(self, step):
        success = self.async_save_engine.save_to_memory(
            step,
            self.ckpt_agent.state_dict,
            self.ckpt_agent.paths,
        )
        return success

    def save_checkpoint_to_storage(self, step):
        success = self.async_save_engine.save_to_storage(
            step,
            self.ckpt_agent.state_dict,
            self.ckpt_agent.paths,
        )
        return success


class HfDeepSpeedCheckpointer(HfFlashCheckpointer):
    def __init__(
        self,
        engine: DeepSpeedEngine,
        checkpoint_dir,
        storage=None,
        comm_backend="",
    ):
        super().__init__(checkpoint_dir, storage)
        self.engine = engine
        global_shard_num = 1
        if self.engine.zero_optimization():
            global_shard_num = dist.get_world_size(
                self.engine.optimizer.dp_process_group
            )
        zero_stage = self.engine.zero_optimization_stage()
        self.async_save_engine = DeepSpeedCheckpointEngine(
            checkpoint_dir,
            storage=self.storage,
            global_shard_num=global_shard_num,
            zero_stage=zero_stage,
            comm_backend=comm_backend,
        )


class HfDdpCheckpointer(HfFlashCheckpointer):
    def __init__(
        self,
        checkpoint_dir,
        storage=None,
        comm_backend="",
    ):
        super().__init__(checkpoint_dir, storage)
        self.async_save_engine = FullCheckpointEngine(
            checkpoint_dir,
            storage=self.storage,
            comm_backend=comm_backend,
        )


class FlashCkptTrainer(Trainer):
    """
    The flash checkpoint trainer synchronously saves the model weights
    and optimizer states of checkpoint into the memory and asynchronously
    saves the checkpoint from the memory to the storage. The training is not
    blocked when saving the checkpoint to the storage.

    Note:: The trainer creates a directory and saves json files of training
    configuration like `config.json`, `trainer_state.json` and
    `generation_config.json` to the directory when saving a checkpoint.
    There might not be model weights and optimizer states in the
    checkpoint directory because the trainer asynchronously save them into the
    directory. We can get the last step of complete checkpoint directory
    by the step int the file `dlrover_latest.txt` in the `OUTPUT_DIR` of
    `TrainingArguments`.
    """

    def _save_checkpoint(self, model, trial, metrics=None):
        run_dir = self._get_output_dir(trial=trial)
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if not hasattr(self, "flash_checkpointer"):
            if self.is_deepspeed_enabled:
                self.flash_checkpointer = HfDeepSpeedCheckpointer(
                    self.model_wrapped, run_dir
                )
            elif not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self.flash_checkpointer = HfDdpCheckpointer(run_dir)
            else:
                raise ValueError(
                    "Flash Checkpoint only supports DeepSpeed or DDP."
                )

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        torch.save = self.flash_checkpointer.ckpt_agent.save
        self.save_model(output_dir, _internal_call=True)
        if self.is_deepspeed_enabled:
            self.model_wrapped.save_checkpoint(output_dir)

        elif (
            self.args.should_save
            and not self.is_deepspeed_enabled
            and not self.is_fsdp_enabled
        ):
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(output_dir, OPTIMIZER_NAME),
            )

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = (
            self.is_deepspeed_enabled
            and not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper)
        )
        if self.args.should_save and (
            not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(
                    self.lr_scheduler.state_dict(),
                    os.path.join(output_dir, SCHEDULER_NAME),
                )
            reissue_pt_warnings(caught_warnings)

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
            self.state.save_to_json(
                os.path.join(output_dir, TRAINER_STATE_NAME)
            )

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global
                # CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to
        # save the model, in which case output_dir may not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(
                rng_states,
                os.path.join(
                    output_dir, f"rng_state_{self.args.process_index}.pth"
                ),
            )
        torch.save = torch_native_save
        success = self.flash_checkpointer.save_checkpoint_to_storage(
            self.state.global_step
        )
        if not success:
            logger.info(
                f"Skip saving the checkpoint of step {self.state.global_step} "
                "because the latest checkpoint is not finished."
            )
            shutil.rmtree(output_dir, ignore_errors=True)

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero
        # so we don't check for that.
        output_dir = (
            output_dir if output_dir is not None else self.args.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (
            (PreTrainedModel,)
            if not is_peft_available()
            else (PreTrainedModel, PeftModel)
        )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if self.args.save_safetensors:
            logger.warn(
                "Flash checkpoint does not support safatensors "
                "and torch.save is used."
            )
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir,
                    state_dict=state_dict,
                    safe_serialization=False,
                    save_function=self.flash_checkpointer.ckpt_agent.save,
                )
            else:
                logger.info(
                    "Trainer.model is not a `PreTrainedModel`, "
                    "only saving its state dict."
                )
                torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=False,
                save_function=self.flash_checkpointer.ckpt_agent.save,
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments
        # together with the trained model
        torch_native_save(
            self.args, os.path.join(output_dir, TRAINING_ARGS_NAME)
        )

    def _rotate_checkpoints(self, use_mtime=False, output_dir=None) -> None:
        if (
            self.args.save_total_limit is None
            or self.args.save_total_limit <= 0
        ):
            return

        last_step = self._get_last_checkpoint_step()

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=use_mtime, output_dir=output_dir
        )

        valid_checkpoints = []
        for path in checkpoints_sorted:
            regex_match = re.match(f".*{PREFIX_CHECKPOINT_DIR}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                step = int(regex_match.groups()[0])
                if step <= last_step:
                    valid_checkpoints.append(path)

        if len(valid_checkpoints) <= self.args.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True,
        # we could end up deleting the last checkpoint, which
        # should be avoided and allow resuming
        save_total_limit = self.args.save_total_limit
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
            and valid_checkpoints[-1] != self.state.best_model_checkpoint
        ):
            save_total_limit = 2

        number_of_checkpoints_to_delete = max(
            0, len(valid_checkpoints) - save_total_limit
        )
        checkpoints_to_be_deleted = valid_checkpoints[
            :number_of_checkpoints_to_delete
        ]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                f"Deleting older checkpoint [{checkpoint}] "
                f"due to save_total_limit = {self.args.save_total_limit}."
            )
            shutil.rmtree(checkpoint, ignore_errors=True)

    def get_last_checkpoint(self):
        """
        Get the path of the last complete checkpoint. Some latter directories
        may not have the complete checkpoint because the asynchronous
        persistence may not finish. The step in the `dlrover_latest.txt` is
        the last step of complete checkpoint. We can get the path by the step.
        """
        step = self._get_last_checkpoint_step()
        if step == 0:
            return False
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{step}"
        ckpt_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        return ckpt_dir

    def _get_last_checkpoint_step(self):
        tracer_file = os.path.join(self.args.output_dir, "dlrover_latest.txt")
        if not os.path.exists(tracer_file):
            return 0
        with open(tracer_file, "r") as f:
            step = int(f.read())
        return step
