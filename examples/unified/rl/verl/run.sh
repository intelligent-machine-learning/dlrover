#!/usr/bin/env bash
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
# This package includes code from [https://github.com/volcengine/verl]
# licensed under the Apache License 2.0. See [https://github.com/volcengine/verl]
# for details.

# Modified from https://github.com/volcengine/verl/blob/main/examples/ppo_trainer/run_deepseek7b_llm.sh
# - Update entrypoint to `main.py`
# - Set `data.dataloader_num_workers=0`, as multiprocessing is not supported well in ray.
# - Removed `wandb` as api-key is not set.
# For quick testing
#   trainer.total_training_steps=3 trainer.save_freq=3

# python -m verl.trainer.main_ppo \
python main.py \
    actor_rollout_ref.model.path=/models/deepseek-ai__deepseek-llm-7b-chat \
    critic.model.path=/models/deepseek-ai__deepseek-llm-7b-chat \
    data.train_files=/ossfs/workspace/data/gsm8k/train.parquet \
    data.val_files=/ossfs/workspace/data/gsm8k/test.parquet \
    data.dataloader_num_workers=0 \
    \
    algorithm.adv_estimator=gae \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=32 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_example_gsm8k' \
    trainer.experiment_name='deepseek_llm_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=1 \
    trainer.use_legacy_worker_impl=auto \
    trainer.total_epochs=15 \
    "$@"
