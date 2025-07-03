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
# licensed under the Apache License 2.0. See [https://github.com/volcengine/
# verl] for details.

import time

import hydra
import ray
from verl.trainer.ppo.ray_trainer import AdvantageEstimator

from dlrover.python.unified.api.rl import RLJobBuilder


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        )

    from pprint import pprint

    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    has_ref = getattr(config.actor_rollout_ref, "init_kl_coef", 0.02)
    has_rew = not getattr(config.trainer, "use_ruled_reward", False)

    if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        has_critic = True
    elif config.algorithm.adv_estimator in [
        AdvantageEstimator.GRPO,
        AdvantageEstimator.REINFORCE_PLUS_PLUS,
        AdvantageEstimator.REMAX,
        AdvantageEstimator.RLOO,
        AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
    ]:
        has_critic = False
    else:
        raise NotImplementedError

    total_node = config["trainer"]["nnodes"]
    device_per_node = config["trainer"]["n_gpus_per_node"]

    rl_job_builder = (
        RLJobBuilder()
        .node_num(total_node)
        .device_per_node(device_per_node)
        .config(config)
        .trainer(
            "dlrover.python.rl.trainer.default.verl.ppo.ppo_trainer",
            "PPOTrainer",
        )
        .actor(
            "dlrover.python.rl.trainer.default.verl.base.megatron_workers",
            "ActorRolloutRefWorker",
        )
        .total(total_node * device_per_node)
        .per_node(device_per_node)
    )
    if has_ref:
        (
            rl_job_builder.reference(
                "dlrover.python.rl.trainer.default.verl.base.megatron_workers",
                "ActorRolloutRefWorker",
            )
            .total(total_node * device_per_node)
            .per_node(device_per_node)
        )
    if has_rew:
        (
            rl_job_builder.reward(
                "dlrover.python.rl.trainer.default.verl.base.megatron_workers",
                "RewardModelWorker",
            )
            .total(total_node * device_per_node)
            .per_node(device_per_node)
        )
    if has_critic:
        (
            rl_job_builder.critic(
                "dlrover.python.rl.trainer.default.verl.base.megatron_workers",
                "CriticWorker",
            )
            .total(total_node * device_per_node)
            .per_node(device_per_node)
        )

    # set colocation
    rl_job_builder.with_collocation_all()

    rl_job = rl_job_builder.build()
    job_name = "dlrover-ppo-" + str(int(time.time()))
    rl_job.submit(job_name=job_name)


if __name__ == "__main__":
    main()
