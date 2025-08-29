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

import multiprocessing
import threading
from concurrent.futures import Future
from functools import wraps
from typing import List, cast

import ray
import verl.workers.fsdp_workers as verl_workers
from omegaconf import DictConfig
from verl.single_controller.base.decorator import MAGIC_ATTR
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.ray.base import func_generator
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.trainer.ppo.reward import load_reward_manager

from dlrover.python.unified.api.runtime.rpc_helper import (
    RoleGroup,
    export_rpc_method,
    rpc,
)
from dlrover.python.unified.api.runtime.worker import current_worker

multiprocessing.set_start_method("spawn", force=True)

end = threading.Event()


def export_rpc(core):
    for f in dir(core):
        v = getattr(core, f)
        if callable(v) and hasattr(v, MAGIC_ATTR):
            export_rpc_method(f, v)


@rpc()
def job_end():
    end.set()


class ActorWorker:
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        role = current_worker().actor_info.role
        self.core = verl_workers.ActorRolloutRefWorker(
            config=config.actor_rollout_ref,
            role=role,
            profile_option=config.trainer.npu_profile.options,
        )
        export_rpc(self.core)

    def run(self):
        end.wait()


class CriticWorker:
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        self.core = verl_workers.CriticWorker(
            config=config.critic,
        )
        export_rpc(self.core)

    def run(self):
        end.wait()


class RMWorker:
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        self.core = verl_workers.RewardModelWorker(config.reward_model)
        export_rpc(self.core)

    def run(self):
        end.wait()


class Trainer:
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)

        "verl.trainer.main_ppo.TaskRunner.run"  # Modified from

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
        "verl.trainer.ppo.ray_trainer.RayPPOTrainer.init_workers"  # Modified from
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
        self.prepare()
        self.trainer.fit()


class MyWorkerGroup(RoleGroup, WorkerGroup):
    def __init__(self, role: str, worker_cls: type) -> None:
        RoleGroup.__init__(self, role)
        WorkerGroup._bind_worker_method(
            cast(WorkerGroup, self), worker_cls, func_generator
        )
        self._patch_ray_get()

    @property
    def world_size(self) -> int:
        return len(self.actors)

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.call(method_name, *args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.call_rank0(method_name, *args, **kwargs)

    # Let linters know this class is dynamic
    def __getattr__(self, item):
        return super().__getattribute__(item)

    @staticmethod
    def _patch_ray_get():
        """execute_all returns Future instead ObjectRef, patch to support ray.get"""
        raw_ray_get = ray.get
        if hasattr(raw_ray_get, "_patched_for_future"):
            return

        @wraps(raw_ray_get)
        def wrap_ray_get(obj, *args, **kwargs):
            if isinstance(obj, Future):
                return obj.result()
            return raw_ray_get(obj, *args, **kwargs)

        setattr(wrap_ray_get, "_patched_for_future", True)
        ray.get = wrap_ray_get
