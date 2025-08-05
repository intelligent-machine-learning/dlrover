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

from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils.deepspeed import DeepspeedStrategy


class RewardModelRayActor:
    def init(self, strategy: DeepspeedStrategy, model_path: str):
        self.strategy = strategy
        strategy.setup_distributed()

        assert isinstance(strategy.args, DictConfig)
        model = get_llm_for_sequence_regression(
            model_path,
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

        if TYPE_CHECKING:
            self.model = model
        else:
            self.model = self.strategy.prepare(model, is_rlhf=True)
        self.model.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        attention_mask: torch.Tensor,
        packed_seq_lens=None,
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
