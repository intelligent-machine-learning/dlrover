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
    DeepSpeedSchedulerWrapper,
    ParallelMode,
    reissue_pt_warnings,
)

from dlrover.python.common.storage import PosixDiskStorage
from dlrover.trainer.torch.flash_checkpoint.ddp_engine import (
    DdpCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.deepspeed import (
    AsyncCheckpointAgent,
)
from dlrover.trainer.torch.flash_checkpoint.deepspeed_engine import (
    DeepSpeedCheckpointEngine,
)
from dlrover.trainer.torch.flash_checkpoint.engine import CheckpointEngine

torch_native_save = torch.save
torch_native_load = torch.load


class HfFlashCheckpointer(object):
    def __init__(self, checkpoint_dir, storage=None):
        self.checkpoint_dir = checkpoint_dir
        self.storage = PosixDiskStorage() if not storage else storage
        self.ckpt_agent = AsyncCheckpointAgent(self.storage)
        self.async_save_engine: Optional[CheckpointEngine] = None

    def save_checkpoint_to_memory(self, step):
        self.async_save_engine.save_to_memory(
            step,
            self.ckpt_agent.state_dict,
            self.ckpt_agent.paths,
        )

    def save_checkpoint_to_storage(self, step):
        self.async_save_engine.save_to_storage(
            step,
            self.ckpt_agent.state_dict,
            self.ckpt_agent.paths,
        )


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
        self.async_save_engine = DdpCheckpointEngine(
            checkpoint_dir,
            storage=self.storage,
            comm_backend=comm_backend,
        )


class FlashCkptTrainer(Trainer):
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
            elif not self.is_deepspeed_enabled and not (
                self.fsdp or self.is_fsdp_enabled
            ):
                self.flash_checkpointer = HfDdpCheckpointer(run_dir)

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        torch.save = self.flash_checkpointer.ckpt_agent.save
        self.save_model(output_dir, _internal_call=True)
        if self.is_deepspeed_enabled:
            self.model_wrapped.save_checkpoint(output_dir)

        elif (
            self.args.should_save
            and not self.is_deepspeed_enabled
            and not (self.fsdp or self.is_fsdp_enabled)
        ):
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(
                self.optimizer.state_dict(),
                os.path.join(output_dir, OPTIMIZER_NAME),
            )
        else:
            raise ValueError(
                "Flash Checkpoint only supports DeepSpeed or DDP."
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

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
