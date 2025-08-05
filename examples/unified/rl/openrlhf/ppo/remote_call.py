from concurrent.futures import Future
from contextlib import contextmanager
from typing import Any, List, Optional, Union

import torch


def vllm_wakeup(): ...
def vllm_sleep(): ...
def vllm_generate(prompt_token_ids, params) -> Any: ...
def vllm_sync_setup_process_group(): ...
def vllm_sync_weight_begin(): ...
def vllm_sync_weight_end(): ...


def reference_init(strategy, model_path: str) -> Future: ...
def reference_forward(
    sequences: torch.LongTensor,
    action_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_output=False,
    packed_seq_lens: Optional[List[int]] = None,
) -> Future[torch.Tensor]: ...


def reward_init(strategy, model_path: str) -> Future: ...
def reward_forward(
    sequences: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    packed_seq_lens=None,
    pad_sequence=False,
) -> Future[torch.Tensor]: ...
def reward_forward_batch(
    sequences: List[torch.Tensor],
    attention_mask: List[torch.LongTensor],
    packed_seq_lens=None,
    pad_sequence=False,
) -> Future[torch.Tensor]: ...


def actor_init(
    strategy,
    model_path: str,
    /,
    max_steps: int,
) -> Future: ...
def actor_forward(
    sequences: torch.LongTensor,
    action_mask: Optional[Union[int, List[int]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_seq_lens=None,
) -> Future[torch.Tensor]: ...
def actor_append_experience(experience) -> Future: ...
def actor_train(kl_ctl: float = 0) -> Future: ...
def actor_sync_to_vllm(): ...
def actor_save_model(save_path: str): ...


def critic_init(
    strategy,
    model_path: str,
    /,
    max_steps: int,
) -> Future: ...
def critic_forward(
    sequences: torch.LongTensor,
    action_mask: Optional[Union[int, list[int]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_seq_lens=None,
) -> Future[torch.Tensor]: ...
def critic_append_experience(experience) -> Future: ...
def critic_train() -> Future: ...
def critic_save_model(save_path: str): ...


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
