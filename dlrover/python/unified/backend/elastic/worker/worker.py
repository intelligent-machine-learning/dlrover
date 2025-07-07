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

import os
import shlex
from contextlib import contextmanager
from datetime import timedelta
from threading import Thread
from typing import List

import ray
import ray.train.torch as ray_train
import torch

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.workload_base import ActorBase, WorkerStage
from dlrover.python.unified.util.os_util import get_free_port


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
    """ElasticWorker is a Ray actor that runs an elastic training job.

    This is different from "worker", which run old elastic agent.
    It skips the torch-run, and directly runs the training in process.
    """

    def _setup(self):
        assert self.node_info.spec.backend == "elastic"
        self.world_size = self.node_info.spec.instance_number
        self.world_rank = self.node_info.rank

        self._setup_envs()
        # RayMasterClient.register_master_actor(f"{self.node_info.role}-master")

    def _setup_envs(self):
        """Setup environment variables for the worker."""

        # referenced ray.train.torch.config.py

        os.environ["LOCAL_RANK"] = str(self.node_info.local_rank)
        os.environ["RANK"] = str(self.node_info.rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.node_info.spec.per_group)
        os.environ["WORLD_SIZE"] = str(self.node_info.spec.instance_number)
        os.environ["NODE_RANK"] = str(
            self.node_info.rank // self.node_info.spec.per_group
        )

        device = ray_train.get_device()
        os.environ["ACCELERATE_TORCH_DEVICE"] = str(device)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.set_device(device)

    def self_check(self):
        """Check the worker itself."""
        if not self._update_stage_if(WorkerStage.PENDING, WorkerStage.INIT):
            return  # already in the expected stage

        logger.info(f"[{self.node_info.name}] Running self check.")
        return "Self check passed"

    # region Rendezvous and process group setup

    def get_master_addr(self):
        """Create a c10d store for distributed training.

        Return master address and port."""
        assert self.world_rank == 0, "Only master can create c10d store."
        addr = ray.util.get_node_ip_address()
        port = get_free_port()

        return f"tcp://{addr}:{port}"

    def setup_torch_process_group(self, master_addr: str):
        """Setup the torch process group for distributed training."""
        assert self.node_info.spec.backend == "elastic"
        backend = self.node_info.spec.comm_backend
        timeout = timedelta(seconds=self.node_info.spec.comm_timeout_s)
        if self.world_rank == 0:
            logger.info(
                f"Setting up torch process group with backend={backend}, "
                f"world_rank={self.world_rank}, world_size={self.world_size}, "
                f"init_method={master_addr}"
            )
        # TODO backend specific setup

        torch.distributed.init_process_group(
            backend=backend,
            init_method=master_addr,
            rank=self.world_rank,
            world_size=self.world_size,
            timeout=timeout,
        )

    def destroy_torch_process_group(self):
        """Destroy the torch process group."""
        devices = ray_train.get_devices()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Destroyed torch process group.")

        if torch.cuda.is_available():
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    # endregion

    def run_node_check(self):
        """TODO Implement node check. Before starting."""
        logger.info(f"[{self.node_info.name}] Running node check.")
        logger.info("SKIP, node check is not implemented yet.")

    def start_elastic_job(self):
        """Start the elastic worker. If already started, do nothing."""
        if not self._update_stage_if(WorkerStage.RUNNING, WorkerStage.PENDING):
            return  # already in the expected stage
        logger.info(f"Starting elastic worker {self.node_info.name}.")

        @contextmanager
        def wrap_run():
            try:
                yield
                self._update_stage_force(
                    WorkerStage.FINISHED, WorkerStage.RUNNING
                )
            except Exception:
                logger.error(
                    "Unexpected error occurred while running elastic agent for training",
                    exc_info=True,
                )
                self._update_stage_force(
                    WorkerStage.FAILED, WorkerStage.RUNNING
                )

        Thread(
            target=wrap_run()(self._run_agent),
            daemon=True,
        ).start()

    def _run_agent(self):
        """Run the elastic agent."""
        logger.info(f"[Rank {self.node_info.rank}] Start Runner")
        # set master handle's actor id as master address

        import dlrover.trainer.torch.node_check.nvidia_gpu

        dlrover.trainer.torch.node_check.nvidia_gpu.run()

        logger.info("Done elastic training.")
