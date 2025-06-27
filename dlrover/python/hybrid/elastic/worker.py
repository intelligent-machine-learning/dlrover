import shlex
from threading import Thread
from typing import List

import ray

from dlrover.python.common import env_utils
from dlrover.python.common.constants import NodeEnv, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import RayMasterClient
from dlrover.python.hybrid.common.node_defines import ActorBase, WorkerStage


def extract_args_from_cmd(run_cmd: str) -> List[str]:
    args_list = shlex.split(run_cmd)

    parsed_args = []
    for arg in args_list[1:]:
        if "=" in arg and arg.startswith("--"):
            key, value = arg.split("=", 1)
            parsed_args.extend([key, value])
        else:
            parsed_args.append(arg)

    return parsed_args


@ray.remote
class ElasticWorker(ActorBase):
    def _setup(self):
        assert self.node_info.spec.kind == "elastic"
        self.cmd = self.node_info.spec.cmd

        env_utils.set_env(
            NodeEnv.JOB_NAME,
            "elastic_job",  # or use env_utils.get_env(DLWorkloadEnv.JOB
        )
        # following envs for compatible
        env_utils.set_env(NodeEnv.NODE_ID, self.node_info.rank)
        env_utils.set_env(NodeEnv.NODE_RANK, self.node_info.rank)
        env_utils.set_env(NodeEnv.NODE_TYPE, NodeType.WORKER)
        env_utils.set_env(NodeEnv.POD_NAME, self.node_info.name)

    def self_check(self):
        """Check the worker itself."""
        if not self._update_stage_if(WorkerStage.PENDING, WorkerStage.INIT):
            return  # already in the expected stage

        print("Worker self check")
        return "Self check passed"

    def start_elastic_job(self):
        """Start the elastic worker. If already started, do nothing."""
        if not self._update_stage_if(WorkerStage.RUNNING, WorkerStage.PENDING):
            return  # already in the expected stage
        logger.info(f"Starting elastic worker {self.node_info.name}.")

        def _run():
            try:
                self._run_agent()
                self._update_stage_force(
                    WorkerStage.FINISHED, WorkerStage.RUNNING
                )
            except Exception as e:
                logger.error(
                    "Failed to run elastic agent for training by "
                    f"unexpected error: {e}",
                    exc_info=True,
                )
                self._update_stage_force(
                    WorkerStage.FAILED, WorkerStage.RUNNING
                )

        Thread(
            target=_run,
            daemon=True,
        ).start()

    def _run_agent(self):
        """Run the elastic agent."""
        logger.info(
            f"Run dlrover command in elastic workload: {self.cmd} "
            f"with node-id(rank): {self.node_info.rank}"
        )
        # set master handle's actor id as master address
        master = f"{self.node_info.role}-master"
        RayMasterClient.register_master_actor(ray.get_actor(master))

        # TODO main(extract_args_from_cmd(self.run_cmd))

        logger.info("Done elastic training.")
