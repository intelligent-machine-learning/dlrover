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

import shlex
from threading import Thread
from typing import List

import ray

from dlrover.python.common.constants import NodeEnv, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.workload_defines import (
    ActorBase,
    WorkerStage,
)

from .runner import ElasticRunner


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

        # Envs, will pass to the elastic agent for compatibility.
        self.envs = {
            NodeEnv.JOB_NAME: str(self.job_info.name),
            NodeEnv.NODE_ID: str(self.node_info.rank),
            NodeEnv.NODE_RANK: str(self.node_info.rank),
            NodeEnv.NODE_TYPE: str(NodeType.WORKER),
            NodeEnv.POD_NAME: str(self.node_info.name),
        }
        self.args = [
            "--nnodes",
            f"{self.node_info.spec.instance_number}",
            "--nproc_per_node",
            f"{self.node_info.spec.per_node}",
            *extract_args_from_cmd(self.cmd),
        ]

    def self_check(self):
        """Check the worker itself."""
        if not self._update_stage_if(WorkerStage.PENDING, WorkerStage.INIT):
            return  # already in the expected stage

        print("Worker self check")
        return "Self check passed"

    def run_node_check(self):
        """TODO Implement node check. Before starting."""

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
        logger.info(f"[Rank {self.node_info.rank}] Start Runner: {self.cmd} ")
        # set master handle's actor id as master address
        master = f"{self.node_info.role}-master"

        runner = (
            ray.remote(ElasticRunner)
            .options(runtime_env={"env_vars": self.envs})
            .remote(self.job_info.name, master, self.args)
        )
        ray.get(runner.run.remote())  # type: ignore
        logger.info("Done elastic training.")

        ray.wait([runner.shutdown.remote()])
