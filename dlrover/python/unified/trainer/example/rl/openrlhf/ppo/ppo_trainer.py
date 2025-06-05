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
import argparse
from typing import Callable, List

import ray
import torch
from openrlhf.cli.train_ppo_ray import reward_fn
from openrlhf.utils import get_strategy

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.trainer.rl_trainer import BaseRLTrainer
from dlrover.python.unified.util.config_util import (
    convert_str_values,
    omega_conf_2_args,
)


class PPOTrainer(BaseRLTrainer):
    def init(self):
        config = self._config
        convert_str_values(config, convert_digit=False)

        args = omega_conf_2_args(config)
        strategy = get_strategy(args)
        refs = []

        # init vllm
        if self.rollouts:
            max_len = (
                args.max_len
                if args.max_len
                else args.prompt_max_len + args.generate_max_len
            )

            num_total_actors = len(self.actors) // args.ring_attn_size
            num_rollout = len(self.rollouts)

            for i, rollout_actor in enumerate(self.rollouts):
                seed = args.seed + i
                if num_rollout >= num_total_actors:
                    num_actors = 1
                else:
                    num_actors = num_total_actors // num_rollout + int(
                        i < num_total_actors % num_rollout
                    )

                logger.info(
                    f"Init rollout[{i}] with: seed={seed},"
                    f" num_actors={num_actors}, "
                    f"gpu_utilization={args.vllm_gpu_memory_utilization}, "
                    f"enable_sleep_mode: {args.vllm_enable_sleep}"
                )

                refs.append(
                    rollout_actor.init.remote(
                        model=args.pretrain,
                        enforce_eager=args.enforce_eager,
                        worker_cls="openrlhf.trainer.ray.vllm_worker_wrap."
                        "WorkerWrap",
                        tensor_parallel_size=args.vllm_tensor_parallel_size,
                        seed=seed,
                        distributed_executor_backend="ray",
                        max_model_len=max_len,
                        enable_prefix_caching=args.enable_prefix_caching,
                        dtype="bfloat16",
                        trust_remote_code=True,
                        num_actors=num_actors,
                        gpu_memory_utilization=(
                            args.vllm_gpu_memory_utilization
                        ),
                        num_gpus=1,
                        enable_sleep_mode=args.vllm_enable_sleep,
                    )
                )
        ray.get(refs)
        logger.info("Done rollout(vllm) init")

        if self.references:
            refs.extend(
                self.RG_REFERENCE.init_model_from_pretrained(
                    strategy, args.pretrain
                )
            )

        refs.extend(
            self.RG_ACTOR.init_model_from_pretrained(strategy, args.pretrain)
        )

        if self.rewards:
            # support only 1 reward pretrain for now
            refs.extend(
                self.RG_REWARD.init_model_from_pretrained(
                    strategy, args.reward_pretrain.split(",")[0]
                )
            )
        ray.get(refs)

        logger.info("Done ref + reward model init")

        if args.critic_pretrain:
            max_steps = ray.get(self.actors[0].max_steps.remote())
            refs.extend(
                self.RG_CRITIC.init_model_from_pretrained(
                    strategy, args.critic_pretrain, max_steps
                )
            )
            ray.get(refs)

        logger.info("Done all model init")

    def fit(self):
        args = argparse.Namespace(**self.config)

        # train
        refs = self.async_fit_actor_model(
            args.remote_rm_url, reward_fn=reward_fn
        )
        ray.get(refs)

        # save
        self.RG_ACTOR.async_save_model()
        if args.critic_pretrain and args.save_value_network:
            self.RG_CRITIC.async_save_model()

    def async_fit_actor_model(
        self,
        remote_rm_urls: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
    ):
        """Train actor model.

        Args:
            remote_rm_urls: remote RM APIs.
            reward_fn: reward calculate function, must be specified if using
                multiple reward models.

        Returns:
            List: list of remote object refs.
        """
        assert (
            remote_rm_urls and len(remote_rm_urls) == 1
        ) or reward_fn is not None, (
            "reward_fn must be specified if using multiple reward models"
        )

        critic_actors = self.critics if self.critics else None
        initial_actors = self.references if self.references else None

        refs = []
        # round robin fashion, implement more efficient dispatching strategy.
        for i, actor in enumerate(self.actors):
            critic_actor = (
                critic_actors[i % len(critic_actors)]
                if critic_actors
                else None
            )
            initial_actor = (
                initial_actors[i % len(initial_actors)]
                if initial_actors
                else None
            )

            reward_actors = []
            if not remote_rm_urls:
                reward_actors.append(self.rewards[i % len(self.rewards)])

            refs.append(
                actor.fit.remote(
                    critic_model=critic_actor,
                    initial_model=initial_actor,
                    reward_model=reward_actors,
                    remote_rm_url=remote_rm_urls,
                    reward_fn=reward_fn,
                    vllm_engines=self.rollouts,
                    critic_train_remote=(i < len(critic_actors))
                    if critic_actor
                    else None,
                )
            )

        return refs
