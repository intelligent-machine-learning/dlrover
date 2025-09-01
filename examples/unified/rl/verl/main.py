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

from pprint import pprint

import hydra
from omegaconf import OmegaConf

from dlrover.python.unified.api.builder.base import DLJobBuilder


@hydra.main(
    config_path="pkg://verl/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    # 1. Ensure the config is resolved, raise early if not
    OmegaConf.resolve(config)

    nodes = config.trainer.nnodes
    gpus = config.trainer.n_gpus_per_node

    # 2. Initialize the job builder
    builder = (
        DLJobBuilder()
        .global_env(
            {
                # "VLLM_USE_V1": "1",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_LOGGING_LEVEL": "DEBUG",
            }
        )
        .node_num(nodes)
        .device_per_node(gpus)
        .config(config)
    )
    # 3. Define roles and their resource requirements
    builder.role("actor_rollout").train("workers.ActorWorker").total(gpus)
    builder.role("critic").train("workers.CriticWorker").total(gpus)
    if config.reward_model.enable:
        builder.role("rm").train("workers.RMWorker").total(gpus)
    if (
        config.algorithm.use_kl_in_reward
        or config.actor_rollout_ref.actor.use_kl_loss
    ):
        builder.role("ref").train("workers.ActorWorker").total(gpus)
    builder.role("trainer").run("workers.Trainer").resource(cpu=4)
    # 4. Share gpu
    builder.with_collocation_all("trainer")

    # 5. Build the job
    job = builder.build()
    for workload in job.workloads.values():
        if workload.backend == "elastic":
            # workload.comm_pre_check = False
            # veRL will setup process group itself, DLRover provide envs
            workload.comm_auto_setup_process_group = False
    pprint(job)

    # 6. Submit the job, launch training
    job.submit(job_name="dlrover-verl-ppo")


if __name__ == "__main__":
    main()
