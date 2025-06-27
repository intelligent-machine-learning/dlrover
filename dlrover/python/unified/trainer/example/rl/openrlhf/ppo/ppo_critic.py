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
from typing import List, Optional, Union

import ray
import torch
from dlrover.python.unified.trainer.default.openrlhf.ppo.ppo_base import (
    BasePPORole,
)
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.trainer.ray.ppo_critic import CriticPPOTrainer
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import (
    offload_deepspeed_states,
    reload_deepspeed_states,
)
from transformers.trainer import get_scheduler

from dlrover.python.unified.trainer.workload import trainer_invocation


@ray.remote
class CriticModelRayActor(BasePPORole):
    @trainer_invocation(blocking=False)
    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, pretrain, max_steps
    ):
        args = strategy.args

        self._setup_distributed(strategy)
        critic = get_llm_for_sequence_regression(
            pretrain,
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
                pretrain,
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
            "cosine_with_min_lr",
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
            )

        # prepare models/optimizers...
        (
            self.critic,
            self.critic_optim,
            self.critic_scheduler,
        ) = strategy.prepare(
            (critic, critic_optim, critic_scheduler),
            is_rlhf=True,
        )

        # load checkpoint
        if args.load_checkpoint and os.path.exists(
            os.path.join(args.ckpt_path, "_actor")
        ):
            ckpt_path = os.path.join(args.ckpt_path, "_critic")
            strategy.load_ckpt(self.critic, ckpt_path)
            strategy.print(f"Loaded the checkpoint: {ckpt_path}")

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            self.offload_states()

        # configure Trainer
        # only use wandb at actor model
        strategy.args.use_wandb = False
        self.trainer = CriticPPOTrainer(
            strategy,
            actor=None,
            critic=self.critic,
            reward_model=None,
            initial_model=None,
            ema_model=None,
            actor_optim=None,
            critic_optim=self.critic_optim,
            actor_scheduler=None,
            critic_scheduler=self.critic_scheduler,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, List[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
    ) -> torch.Tensor:
        """Generates critic values."""
        device = torch.cuda.current_device()
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(
                sequences.to(device),
                num_actions,
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                values_allgather=True,
                packed_seq_lens=packed_seq_lens,
            )
        self.critic.train()  # reset model state
        return value.to("cpu")

    def append(self, experience):
        """Append experience to replay buffer."""
        self.trainer.replay_buffer.append(experience)

    def fit(self):
        """Train critic model with the replay buffer."""
        torch.cuda.empty_cache()
        self.critic.train()
        status = self.trainer.ppo_train()
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        return status

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.critic,
            self.tokenizer,
            args.save_path + "_critic",
        )

    def save_checkpoint(self, tag):
        args = self.strategy.args
        self.strategy.save_ckpt(
            self.critic,
            os.path.join(args.ckpt_path, "_critic"),
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
        )

    def reload_states(self):
        reload_deepspeed_states(self.critic)

    def offload_states(self):
        offload_deepspeed_states(self.critic)
