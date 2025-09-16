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

# Multi-Role Training with DLRover
# Simplified/Extracted from examples/unified/rl/openrlhf


from typing import List, Sequence

from omegaconf import DictConfig

from dlrover.python.unified.api.runtime import current_worker, rpc

from . import remote_call


class Actor:
    @rpc(remote_call.actor_init)
    def init(self, *args, **kwargs):
        print(f"Actor initialized with model: {args} {kwargs}")
        self.initialized = True

    @rpc(remote_call.actor_forward)
    def batch_forward(
        self, sequences: Sequence[str], action_mask, attention_mask
    ) -> Sequence:
        assert self.initialized
        print(
            f"Batch forward called with sequences: {sequences}, action_mask: {action_mask}, attention_mask: {attention_mask}"
        )
        return [x[-2:] for x in sequences]

    @rpc(remote_call.actor_append_experience)
    def append_experience(self, experience: List[str]):
        assert self.initialized
        print(f"Experience appended: {experience}")
        self.experience = experience

    @rpc(remote_call.actor_train)
    def fit(self, *args, **kwargs):
        assert self.experience
        print(f"Actor fitted with experience: {self.experience}")
        self.fitted = True
        return {"actor_loss": 0.1}

    @rpc(remote_call.actor_save_model)
    def save_checkpoint(self, checkpoint_path: str, *args, **kwargs):
        assert self.fitted, "Actor must be fitted before saving checkpoint"
        print(f"Checkpoint saved at: {checkpoint_path}")


class Critic:
    @rpc(remote_call.critic_init)
    def init(self, *args, **kwargs):
        print(f"Critic initialized with model: {args} {kwargs}")
        self.initialized = True

    @rpc(remote_call.critic_forward)
    def batch_forward(
        self, sequences: Sequence[str], action_mask, attention_mask
    ) -> Sequence:
        assert self.initialized
        print(f"Batch forward called with sequences: {sequences}")
        return [x * 2 for x in sequences]

    @rpc(remote_call.critic_append_experience)
    def append_experience(self, experience: List[str]):
        assert self.initialized
        print(f"Experience appended: {experience}")
        self.experience = experience

    @rpc(remote_call.critic_train)
    def fit(self, *args, **kwargs):
        assert self.experience
        print(f"Critic fitted with experience: {self.experience}")
        self.fitted = True
        return {"critic_loss": 0.2}

    @rpc(remote_call.critic_save_model)
    def save_checkpoint(self, checkpoint_path: str, *args, **kwargs):
        assert self.fitted, "Critic must be fitted before saving checkpoint"
        print(f"Checkpoint saved at: {checkpoint_path}")


class Reference:
    @rpc(remote_call.reference_init)
    def init(self, model_path: str, *args, **kwargs):
        print(f"Reference initialized with model: {model_path}")
        self.initialized = True

    @rpc(remote_call.reference_forward)
    def batch_forward(
        self, sequences: Sequence[str], *args, **kwargs
    ) -> Sequence:
        assert self.initialized
        print(f"Batch forward called with sequences: {sequences}")
        return [x * 2 for x in sequences]


class Reward:
    @rpc(remote_call.reward_init)
    def init(self, model_path: str, *args, **kwargs):
        print(f"Reward initialized with model: {model_path}")
        self.initialized = True

    @rpc(remote_call.reward_forward)
    def batch_forward(
        self, sequences: Sequence[str], *args, **kwargs
    ) -> Sequence:
        assert self.initialized
        print(f"Batch forward called with sequences: {sequences}")
        return [x * 2 for x in sequences]


class Rollout:
    @rpc(remote_call.vllm_generate)
    def generate(self, input: Sequence[str], *args, **kwargs) -> Sequence:
        print(f"Rollout generated with input: {input}")
        return [f"<generated_text {x}>" for x in input]


class Trainer:
    def __init__(self):
        config = DictConfig(current_worker().job_info.user_config)
        self.config = config

    def run(self):
        self.prepare_datasets()
        self.init_workers()

        self.fit()

    def prepare_datasets(self):
        print("Preparing datasets...")
        self.max_steps = 999

    def init_workers(self):
        print("Initializing workers...")

        futures = [
            remote_call.reference_init("strategy", "args.pretrain"),
            remote_call.reward_init("strategy", "args.reward_pretrain"),
            remote_call.actor_init(
                "strategy", "args.pretrain", max_steps=self.max_steps
            ),
            remote_call.critic_init(
                "strategy",
                "args.critic_pretrain",
                max_steps=self.max_steps,
            ),
        ]
        for future in futures:
            future.result()

    def fit(self):
        for episode in range(3):
            prompts = [f"Prompt {episode}_{i}" for i in range(5)]
            # generate_samples
            outputs = remote_call.vllm_generate(prompts, 0)
            experiences = [
                f"P: {it[0]}\nA: {it[1]}" for it in zip(prompts, outputs)
            ]
            # make_experience_batch
            rewards = remote_call.reward_forward(
                experiences,
                experiences,
            )
            log_probs = remote_call.actor_forward(
                experiences,
                experiences,
                experiences,
            )
            values = remote_call.critic_forward(
                experiences,
                experiences,
                experiences,
            )
            ref_log_probs = remote_call.reference_forward(
                experiences,
                experiences,
                experiences,
            )
            for i in range(len(experiences)):
                experiences[i] += (
                    f"\nR: {rewards[i]}, LP: {log_probs[i]}, V: {values[i]}, RLP: {ref_log_probs[i]}"
                )
            print(f"Sample experiences: {experiences[0]}")
            # append experiences to remote actor and critic
            futures = [
                remote_call.actor_append_experience(experiences),
                remote_call.critic_append_experience(experiences),
            ]
            [it.result() for it in futures]  # wait
            # ppo_train
            status = {}
            critic_status_ref = remote_call.critic_train()
            actor_status_ref = remote_call.actor_train()
            status.update(actor_status_ref.result()[0])
            status.update(critic_status_ref.result()[0])
            # save_logs_and_checkpoints
            print(f"Episode {episode} training status: {status}")
            futures = [
                remote_call.actor_save_model("args.ckpt_path" + "_actor"),
                remote_call.critic_save_model("args.ckpt_path" + "_critic"),
            ]
            [it.result() for it in futures]  # wait
        print("Training completed.")
