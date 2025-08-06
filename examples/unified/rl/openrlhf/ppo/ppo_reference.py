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

from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
from omegaconf import DictConfig
from openrlhf.models import Actor
from openrlhf.utils.deepspeed import DeepspeedStrategy

from dlrover.python.unified.api.runtime.rpc import rpc
from examples.unified.rl.openrlhf.ppo import remote_call


class PPOReferenceActor:
    @rpc(remote_call.reference_init)
    def init(self, strategy: DeepspeedStrategy, model_path: str):
        self.strategy = strategy
        strategy.setup_distributed()

        assert isinstance(strategy.args, DictConfig)
        model = Actor(
            model_path,
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

        if TYPE_CHECKING:
            self.model = model
        else:
            self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_lens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.model(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
                packed_seq_lens=packed_seq_lens,
            )
        return log_probs.to("cpu")

    @rpc(remote_call.reference_forward)
    def batch_forward(
        self,
        sequences: Sequence[torch.LongTensor],
        action_mask: Sequence[torch.Tensor],
        attention_mask: Sequence[torch.Tensor],
        packed_seq_lens: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        return [
            self.forward(*args, packed_seq_lens=packed_seq_lens)
            for args in zip(sequences, action_mask, attention_mask)
        ]

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()
