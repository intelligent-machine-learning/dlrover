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
    OmegaConf.resolve(config)

    nodes = config.trainer.nnodes
    gpus = config.trainer.n_gpus_per_node

    builder = (
        DLJobBuilder()
        .node_num(nodes)
        .device_per_node(gpus)
        .config(config)
        .device_type("CPU")
    )
    builder.role("actor_rollout").train("workers.ActorWorker").nnodes(
        nodes
    ).nproc_per_node(gpus)
    builder.role("critic").train("workers.CriticWorker").nnodes(
        nodes
    ).nproc_per_node(gpus)
    if config.reward_model.enable:
        builder.role("rm").train("workers.RMWorker").nnodes(
            nodes
        ).nproc_per_node(gpus)
    if (
        config.algorithm.use_kl_in_reward
        or config.actor_rollout_ref.actor.use_kl_loss
    ):
        builder.role("ref").train("workers.ActorWorker").nnodes(
            nodes
        ).nproc_per_node(gpus)
    builder.role("trainer").run("workers.Trainer").resource(cpu=4)
    builder.with_collocation_all("trainer")

    job = builder.build()
    for workload in job.workloads.values():
        if workload.backend == "elastic":
            workload.comm_pre_check = False
            workload.comm_auto_setup_process_group = False

    pprint(job)
    job.submit(job_name="dlrover-verl-ppo")


if __name__ == "__main__":
    main()
