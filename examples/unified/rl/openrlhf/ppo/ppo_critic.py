# Copyright 2025 The DLRover Authors. All rights reserved.
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
#
# This package includes code from [https://github.com/OpenRLHF/OpenRLHF]
# licensed under the Apache License 2.0. See [https://github.com/OpenRLHF/
# OpenRLHF] for details.

import math
import os
from typing import TYPE_CHECKING, Sequence

import torch
from omegaconf import DictConfig
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer.ray.ppo_critic import CriticPPOTrainer
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import (
    offload_deepspeed_states,
    reload_deepspeed_states,
)
from transformers.optimization import get_scheduler

from . import remote_call
from .common import BaseActor, rpc


class CriticModelRayActor(BaseActor):
    @rpc(remote_call.critic_init)
    def init(
        self,
        strategy: DeepspeedStrategy,
        model_path: str,
        /,
        max_steps: int,
    ):
        self.strategy = strategy
        strategy.setup_distributed()
        assert isinstance(strategy.args, DictConfig)
        args = strategy.args
        self.args = args

        critic = get_llm_for_sequence_regression(
            model_path,
            "critic",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            value_head_prefix=strategy.args.value_head_prefix,
            init_value_head=strategy.args.pretrain
            == strategy.args.critic_pretrain,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(critic)
        strategy.print(
            "reward normalization status: {}".format(
                strategy.args.normalize_reward
            )
        )
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

        # configure tokenizer
        if strategy.args.save_value_network:
            self.tokenizer = get_tokenizer(
                model_path,
                critic,
                "left",
                strategy,
                use_fast=not strategy.args.disable_fast_tokenizer,
            )

        # configure optimizer
        critic_optim = strategy.create_optimizer(
            critic,
            lr=args.critic_learning_rate,
            betas=args.adam_betas,
            weight_decay=args.l2,
        )

        # configure scheduler
        critic_scheduler = get_scheduler(
            args.lr_scheduler,
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={
                "min_lr": args.critic_learning_rate * 0.1
            },
        )

        if args.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )  # type: ignore[call-arg]

        # prepare models/optimizers...
        if TYPE_CHECKING:
            self.critic, self.critic_optim, self.critic_scheduler = (
                critic,
                critic_optim,
                critic_scheduler,
            )
        else:
            self.critic, self.critic_optim, self.critic_scheduler = (
                strategy.prepare(
                    (critic, critic_optim, critic_scheduler),
                    is_rlhf=True,
                )
            )

        # load checkpoint
        if args.load_checkpoint and os.path.exists(
            os.path.join(args.ckpt_path, "_actor")
        ):
            ckpt_path = os.path.join(args.ckpt_path, "_critic")
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            strategy.load_ckpt(self.critic, ckpt_path)

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.critic)

        # configure Trainer
        self.trainer = CriticPPOTrainer(
            strategy,
            critic=self.critic,
            critic_optim=self.critic_optim,
            critic_scheduler=self.critic_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            value_clip=args.value_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: torch.BoolTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                values_allgather=True,
            )
        self.critic.train()  # reset model state
        return value.to("cpu")

    @rpc(remote_call.critic_forward)
    def batch_forward(
        self,
        sequences: Sequence[torch.LongTensor],
        action_mask: Sequence[torch.BoolTensor],
        attention_mask: Sequence[torch.LongTensor],
    ) -> list[torch.Tensor]:
        return [
            self.forward(*args)
            for args in zip(sequences, action_mask, attention_mask)
        ]

    @rpc(remote_call.critic_append_experience)
    def append(self, experience):
        """Append experience to replay buffer."""
        self.trainer.replay_buffer.append(experience)

    @rpc(remote_call.critic_train)
    def fit(self):
        """Train critic model with the replay buffer."""
        if self.args.deepspeed_enable_sleep:
            reload_deepspeed_states(self.critic)
        torch.cuda.empty_cache()
        self.critic.train()
        status = self.trainer.ppo_train()
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if self.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.critic)
        return status

    @rpc(remote_call.critic_save_model)
    def save_checkpoint(self, save_path: str, tag):
        args = self.args
        self.strategy.save_ckpt(
            self.critic,
            save_path,
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
        )
