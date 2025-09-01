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

from concurrent.futures import Future
from typing import List

import verl.workers.fsdp_workers as verl_workers
from omegaconf import DictConfig
from util import (
    BaseWorker,
    MyWorkerGroup,
    notify_job_end,
)
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.trainer.ppo.reward import load_reward_manager

from dlrover.python.unified.api.runtime.worker import current_worker


class ActorWorker(BaseWorker):
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        role = current_worker().actor_info.role
        self.core = verl_workers.ActorRolloutRefWorker(
            config=config.actor_rollout_ref,
            role=role,
            profile_option=config.trainer.npu_profile.options,
        )
        super().__init__(self.core)


class CriticWorker(BaseWorker):
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        self.core = verl_workers.CriticWorker(
            config=config.critic,
        )
        super().__init__(self.core)


class RMWorker(BaseWorker):
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        self.core = verl_workers.RewardModelWorker(config.reward_model)
        super().__init__(self.core)


class Trainer:
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)

        # Modified from verl.trainer.main_ppo.TaskRunner.run

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(
            local_path, trust_remote_code=trust_remote_code
        )
        processor = hf_processor(
            local_path, trust_remote_code=trust_remote_code, use_fast=True
        )

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # We skip trainer.init_workers, we mock some to keep it works.
        # Mock objects
        class FakeResourcePoolManager:
            def get_n_gpus(self):
                return config.trainer.n_gpus_per_node

        roles = [Role.ActorRollout, Role.Critic]
        if config.reward_model.enable:
            roles.append(Role.RewardModel)
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            roles.append(Role.RefPolicy)

        self.trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping={role: None for role in roles},  # type: ignore
            resource_pool_manager=FakeResourcePoolManager(),  # type: ignore
            ray_worker_group_cls=None,  # type: ignore
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

    def prepare(self):
        # Modified from RayPPOTrainer.init_workers
        trainer = self.trainer
        async_init: List[Future] = []
        if trainer.use_critic:
            trainer.critic_wg = MyWorkerGroup(
                "critic", verl_workers.CriticWorker
            )
            async_init.append(trainer.critic_wg.call("init_model"))

        if trainer.use_reference_policy and not trainer.ref_in_actor:
            trainer.ref_policy_wg = MyWorkerGroup(
                "ref", verl_workers.ActorRolloutRefWorker
            )
            async_init.append(trainer.ref_policy_wg.call("init_model"))

        if trainer.use_rm:
            trainer.rm_wg = MyWorkerGroup("rm", verl_workers.RewardModelWorker)
            async_init.append(trainer.rm_wg.call("init_model"))

        # We init workers in parallel
        [it.result() for it in async_init]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        trainer.actor_rollout_wg = MyWorkerGroup(
            "actor_rollout", verl_workers.AsyncActorRolloutRefWorker
        )
        trainer.actor_rollout_wg.init_model()

        trainer.async_rollout_mode = False
        if trainer.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            trainer.async_rollout_mode = True
            trainer.async_rollout_manager = AgentLoopManager(
                config=trainer.config,
                worker_group=trainer.actor_rollout_wg,  # type: ignore
            )

    def run(self):
        try:
            self.prepare()
            self.trainer.fit()
        finally:
            notify_job_end("critic", "ref", "rm", "actor_rollout")
