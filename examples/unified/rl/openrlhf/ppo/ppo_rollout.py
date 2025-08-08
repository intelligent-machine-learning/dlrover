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

import os
from dataclasses import dataclass
from typing import Optional

import vllm
from omegaconf import OmegaConf
from openrlhf.utils.logging_utils import init_logger

from dlrover.python.unified.api.runtime.worker import current_worker

from . import remote_call
from .common import BaseActor, rpc

logger = init_logger(__name__)


def _vllm_setup_process_group(
    self, rank_start, world_size, backend, group_name
):
    import ray.util.collective as collective
    import torch

    collective.init_collective_group(
        world_size=world_size,
        rank=rank_start + torch.distributed.get_rank(),
        backend=backend,
        group_name=group_name,
    )
    self._model_update_group = group_name


def _vllm_update_weight(self, name, dtype, shape):
    import ray.util.collective as collective
    import torch

    assert dtype == self.model_config.dtype, (
        f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
    )
    weight = torch.empty(shape, dtype=dtype, device="cuda")
    collective.broadcast(weight, 0, group_name=self._model_update_group)
    self.model_runner.model.load_weights(weights=[(name, weight)])
    del weight


@dataclass(kw_only=True)
class VLLMActorConfig:
    pretrain: str
    full_determinism: bool = False
    enforce_eager: bool = False
    vllm_tensor_parallel_size: int = 1
    seed: int = 0
    max_len: Optional[int] = 2048
    prompt_max_len: int
    generate_max_len: int
    enable_prefix_caching: bool = False
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enable_sleep: bool = True


class VLLMActor(BaseActor):
    def __init__(self):
        super().__init__()
        config: VLLMActorConfig = OmegaConf.structured(VLLMActorConfig)
        OmegaConf.unsafe_merge(
            config,
            OmegaConf.masked_copy(
                OmegaConf.create(current_worker().job_info.user_config),
                dir(config),
            ),
        )
        self.config = config

        rank = current_worker().actor_info.rank

        if config.full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        if vllm.__version__ >= "0.9.0":
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        tp = config.vllm_tensor_parallel_size
        max_len = (
            config.max_len
            if config.max_len
            else config.prompt_max_len + config.generate_max_len
        )
        self.llm = vllm.LLM(
            config.pretrain,
            enforce_eager=config.enforce_eager,
            tensor_parallel_size=tp,
            seed=config.seed + rank,
            distributed_executor_backend="uni" if tp == 1 else "ray",
            max_model_len=max_len,
            enable_prefix_caching=config.enable_prefix_caching,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        )

        if config.vllm_enable_sleep:
            self.sleep()

    @rpc(remote_call.vllm_sync_setup_process_group)
    def init_process_group(
        self,
        world_size,
        backend,
        group_name,
    ):
        tp = self.config.vllm_tensor_parallel_size
        assert world_size == current_worker().actor_info.spec.total * tp + 1, (
            f"world_size {world_size} mismatch with total {current_worker().actor_info.spec.total} and vllm_tensor_parallel_size {tp}"
        )
        rank_start = current_worker().actor_info.rank * tp + 1
        return self.llm.collective_rpc(
            _vllm_setup_process_group,
            args=(rank_start, world_size, backend, group_name),
        )

    @rpc(remote_call.vllm_sync_weight_begin)
    def sync_weight_begin(self):
        if self.config.enable_prefix_caching:
            self.llm.llm_engine.reset_prefix_cache()

    @rpc(remote_call.vllm_sync_weight)
    def update_weight(self, name, dtype, shape):
        return self.llm.collective_rpc(
            _vllm_update_weight, args=(name, dtype, shape)
        )

    @rpc(remote_call.vllm_sync_weight_end)
    def sync_weight_end(self):
        pass

    @rpc(remote_call.vllm_sleep)
    def sleep(self, level=1):
        self.llm.sleep(level=level)

    @rpc(remote_call.vllm_wakeup)
    def wake_up(self):
        self.llm.wake_up()

    @rpc(remote_call.vllm_generate)
    def generate(self, prompt_token_ids, params):
        """
        Generate responses for the given prompt token IDs using the provided sampling parameters.
        """

        requests = [
            vllm.TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids
        ]
        responses = self.llm.generate(requests, sampling_params=params)
        return responses
