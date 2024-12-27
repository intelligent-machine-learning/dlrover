# Copyright 2024 The DLRover Authors. All rights reserved.
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

export XPU_TIMER_HOST_TRACING_FUNC=transformers.models.gpt_neox.modeling_gpt_neox@GPTNeoXAttention@forward,transformers.models.gpt_neox.modeling_gpt_neox@GPTNeoXRotaryEmbedding@forward,transformers.models.gpt_neox.modeling_gpt_neox@GPTNeoXMLP@forward,transformers.models.gpt_neox.modeling_gpt_neox@GPTNeoXLayer@forward
xpu_timer_launch python -m torch.distributed.launch --nproc_per_node=8 train_fsdp.py
