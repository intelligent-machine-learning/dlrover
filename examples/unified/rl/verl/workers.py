from concurrent.futures import Future
from typing import List, cast

import verl.workers.fsdp_workers as verl_workers
from omegaconf import DictConfig
from verl.protocol import DataProto, _padding_size_key
from verl.single_controller.base.worker_group import WorkerGroup
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role
from verl.trainer.ppo.reward import load_reward_manager

from dlrover.python.unified.api.runtime.rpc_helper import (
    RoleGroup,
    export_rpc_instance,
)
from dlrover.python.unified.api.runtime.worker import current_worker


class ActorWorker:
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        role = current_worker().actor_info.role
        self.core = verl_workers.ActorRolloutRefWorker(
            config=config.actor_rollout_ref,
            role=role,
            profile_option=config.trainer.npu_profile.options,
        )
        export_rpc_instance(None, self.core)


class CriticWorker:
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        self.core = verl_workers.CriticWorker(
            config=config.actor_rollout_ref,
        )
        export_rpc_instance(None, self.core)


class RMWorker:
    def __init__(self) -> None:
        config = DictConfig(current_worker().job_info.user_config)
        self.core = verl_workers.RewardModelWorker(config.reward_model)
        export_rpc_instance(None, self.core)


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
                return 0

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

    def run(self):
        self.prepare()
        self.trainer.fit()


def func_generator(
    self, method_name, dispatch_fn, collect_fn, execute_fn, blocking
):
    assert blocking, "Blocking must be True"

    class Functor:
        @staticmethod
        def __call__(*args, **kwargs):
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)
            output = collect_fn(self, output)
            if padding_count > 0:
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count]
                    output = output.select_idxs(indices)
                elif isinstance(output, list):
                    output = output[:-padding_count]
            return output

    # use class type to pass the method_name to get a better observability
    return type(method_name, (Functor,), {})()


class MyWorkerGroup(RoleGroup):
    def __init__(self, role: str, worker_cls: type) -> None:
        RoleGroup.__init__(self, role)
        WorkerGroup._bind_worker_method(
            cast(WorkerGroup, self), worker_cls, func_generator
        )

    def execute_all(self, method_name: str, *args, **kwargs):
        length = len(self.actors)
        if all(isinstance(arg, list) for arg in args) and all(
            isinstance(kwarg, list) for kwarg in kwargs.values()
        ):
            if all(len(arg) == length for arg in args) and all(
                len(kwarg) == length for kwarg in kwargs.values()
            ):
                # print(f"splitting args and kwargs into {length} shards")
                result: List[Future] = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(
                        self.actors[i].call(
                            method_name,
                            *sliced_args,
                            **sliced_kwargs,
                        )
                    )
                return [r.result() for r in result]
        return self.call(method_name, *args, **kwargs).result()

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.call_rank0(method_name, *args, **kwargs).result()

    # Let linters know this class is dynamic
    def __getattr__(self, item):
        return super().__getattribute__(item)
