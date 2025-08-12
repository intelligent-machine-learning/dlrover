#!/usr/bin/env bash
# -*- coding: utf-8 -*-
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

if [ $INSTALL_DEPS ]; then
    pip install .[torch,ray] #dlrover
    pip install flash-attn==2.8.2 --no-build-isolation
    pip install openrlhf	
    pip install cupy-cuda12x #ray collective
fi

if [ $USE_MS ]; then
    pip install modelscope
    modelscope download AI-ModelScope/Llama-3-8b-sft-mixture
    modelscope download AI-ModelScope/Llama-3-8b-rm-mixture
    modelscope download --dataset AI-ModelScope/prompt-collection-v0.1
fi

python3 -m rl.openrlhf.ppo.main \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --colocate_critic_reward \
   --colocate_actor_ref \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path /openrlhf/examples/checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 16 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 1024 \
   --max_samples 100000 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 1 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   "$@"

#    --pretrain /root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3-8b-sft-mixture \
#    --reward_pretrain /root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3-8b-rm-mixture \
#    --prompt_data /root/.cache/modelscope/hub/datasets/AI-ModelScope/prompt-collection-v0___1 \
#    --use_wandb {wandb_token}
