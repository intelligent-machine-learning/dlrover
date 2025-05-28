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

import socket
from typing import List, Optional

import ray
import torch
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.utils.deepspeed import DeepspeedStrategy

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.trainer.rl_workload import BaseRLWorkload
from dlrover.python.unified.trainer.workload import (
    trainer_invocation,
)


class BasePPORole(BaseRLWorkload):
    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self.torch_master_addr, self.torch_master_port

    def _setup_distributed(self, strategy: DeepspeedStrategy):
        # configure strategy
        self.strategy = strategy
        strategy.setup_distributed()
        logger.info(
            f"{self.name} done dist init, "
            f"using device: {torch.cuda.current_device()}"
        )

    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()


@ray.remote
class ReferenceModelRayActor(BasePPORole):
    @trainer_invocation(blocking=False)
    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, pretrain
    ):
        self._setup_distributed(strategy)
        model = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(
                offload=strategy.args.ref_reward_offload
            ),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(model)

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        logps_allgather=False,
        packed_seq_lens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                num_actions,
                attention_mask.to(device),
                return_output=return_output,
                ring_attn_group=self.strategy.ring_attn_group,
                logps_allgather=logps_allgather,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()


@ray.remote
class RewardModelRayActor(BasePPORole):
    @trainer_invocation(blocking=False)
    def init_model_from_pretrained(
        self, strategy: DeepspeedStrategy, pretrain
    ):
        self._setup_distributed(strategy)
        model = get_llm_for_sequence_regression(
            pretrain,
            "reward",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            ds_config=strategy.get_ds_eval_config(
                offload=strategy.args.ref_reward_offload
            ),
            value_head_prefix=strategy.args.value_head_prefix,
            packing_samples=strategy.args.packing_samples,
        )
        strategy.print(model)
        strategy.print(
            "reward normalization status: {}".format(
                strategy.args.normalize_reward
            )
        )
        strategy.print("mean: {}, std {}".format(model.mean, model.std))

        if strategy.args.ref_reward_offload:
            model._offload = True

        self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_lens=None,
        pad_sequence=False,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            reward = self.model(
                sequences.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                pad_sequence=True,
                packed_seq_lens=packed_seq_lens,
            )
        return reward.to("cpu")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
