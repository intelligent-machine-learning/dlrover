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
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, List, Optional, Sequence, TypeAlias

from dlrover.python.unified.api.runtime import FutureSequence, RoleGroup
from dlrover.python.unified.common.enums import RLRoleType

# Mock
SamplingParams: TypeAlias = Any
RequestOutput: TypeAlias = Any
Experience: TypeAlias = Any
Tensor: TypeAlias = Any


# endregion
# region Rollout


def vllm_wakeup() -> None:
    group(RLRoleType.ROLLOUT).call(vllm_wakeup).result()


def vllm_sleep() -> None:
    group(RLRoleType.ROLLOUT).call(vllm_sleep).result()


def vllm_generate(
    prompt_token_ids: Sequence[str], params: "SamplingParams"
) -> FutureSequence["RequestOutput"]:
    return group(RLRoleType.ROLLOUT).call_batch(
        vllm_generate,
        len(prompt_token_ids),
        prompt_token_ids,
        params,
    )


def vllm_sync_setup_process_group(
    world_size: int, backend: str, group_name: str
) -> Future:
    return group(RLRoleType.ROLLOUT).call(
        vllm_sync_setup_process_group, world_size, backend, group_name
    )


def vllm_sync_weight_begin():
    group(RLRoleType.ROLLOUT).call(vllm_sync_weight_begin).result()


def vllm_sync_weight(name: str, dtype, shape) -> Future:
    return group(RLRoleType.ROLLOUT).call(vllm_sync_weight, name, dtype, shape)


def vllm_sync_weight_end():
    group(RLRoleType.ROLLOUT).call(vllm_sync_weight_end).result()


# endregion
# region Reference


def reference_init(strategy, model_path: str) -> Future:
    return group(RLRoleType.REFERENCE).call(
        reference_init, strategy, model_path
    )


def reference_forward(
    sequences: Sequence["Tensor"],
    action_mask: Sequence["Tensor"],
    attention_mask: Sequence["Tensor"],
    packed_seq_lens: Optional[Sequence[int]] = None,
) -> Sequence["Tensor"]:
    return group(RLRoleType.REFERENCE).call_batch(
        reference_forward,
        len(sequences),
        sequences,
        action_mask,
        attention_mask,
        packed_seq_lens,
    )


# endregion
# region Reward


def reward_init(strategy, model_path: str) -> Future:
    return group(RLRoleType.REWARD).call(reward_init, strategy, model_path)


def reward_forward(
    sequences: Sequence["Tensor"],
    attention_mask: Sequence["Tensor"],
    packed_seq_lens=None,
) -> Sequence["Tensor"]:
    return group(RLRoleType.REWARD).call_batch(
        reward_forward,
        len(sequences),
        sequences,
        attention_mask,
        packed_seq_lens,
    )


# endregion
# region Actor


def actor_init(
    strategy,
    model_path: str,
    /,
    max_steps: int,
) -> Future:
    return group(RLRoleType.ACTOR).call(
        actor_init, strategy, model_path, max_steps=max_steps
    )


def actor_forward(
    sequences: Sequence["Tensor"],
    action_mask: Sequence["Tensor"],
    attention_mask: Sequence["Tensor"],
) -> Sequence["Tensor"]:
    return group(RLRoleType.ACTOR).call_batch(
        actor_forward, len(sequences), sequences, action_mask, attention_mask
    )


def actor_append_experience(experience: List["Experience"]) -> Future:
    return group(RLRoleType.ACTOR).call(actor_append_experience, experience)


def actor_train(kl_ctl: float = 0) -> Future:
    """Train the actor with the given KL control value."""
    return group(RLRoleType.ACTOR).call(actor_train, kl_ctl)


def actor_sync_to_vllm() -> None:
    """Synchronize the actor's weights to vLLM."""
    group(RLRoleType.ACTOR).call(actor_sync_to_vllm).result()


def actor_save_model(
    save_path: str,
    tag: Optional[str] = None,
    ext_states: Optional[dict] = None,
) -> Future:
    """Save the actor's model to the specified path."""
    return group(RLRoleType.ACTOR).call(
        actor_save_model, save_path, tag, ext_states
    )


# endregion
# region Critic


def critic_init(
    strategy,
    model_path: str,
    /,
    max_steps: int,
) -> Future:
    return group(RLRoleType.CRITIC).call(
        critic_init, strategy, model_path, max_steps=max_steps
    )


def critic_forward(
    sequences: Sequence["Tensor"],
    action_mask: Sequence["Tensor"],
    attention_mask: Sequence["Tensor"],
) -> Sequence["Tensor"]:
    return group(RLRoleType.CRITIC).call_batch(
        critic_forward, len(sequences), sequences, action_mask, attention_mask
    )


def critic_append_experience(experience: List["Experience"]) -> Future:
    return group(RLRoleType.CRITIC).call(critic_append_experience, experience)


def critic_train() -> Future:
    """Train the critic."""
    return group(RLRoleType.CRITIC).call(critic_train)


def critic_save_model(save_path: str, tag: Optional[str] = None) -> Future:
    """Save the critic's model to the specified path."""
    return group(RLRoleType.CRITIC).call(critic_save_model, save_path, tag)


# endregion
# region Utility


@lru_cache(maxsize=None)
def group(role: RLRoleType) -> RoleGroup:
    """Get the role group for the given role."""
    return RoleGroup(role.name)


@contextmanager
def vllm_running(enable_sleep: bool):
    """Context manager to handle vLLM sleep and wakeup."""
    if enable_sleep:
        vllm_wakeup()
    try:
        yield
    finally:
        if enable_sleep:
            vllm_sleep()
