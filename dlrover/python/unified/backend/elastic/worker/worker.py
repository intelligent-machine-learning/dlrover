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
from datetime import timedelta

import ray
import torch

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.backend.elastic.events import ElasticWorkerEvents
from dlrover.python.unified.common.enums import (
    ACCELERATOR_TYPE,
    ExecutionResult,
)
from dlrover.python.util.common_util import (
    find_free_port_from_env_and_bind,
)


def _get_ray_gpu_devices():
    """Get the devices assigned to this worker."""
    gpu_ids = [str(id) for id in ray.get_gpu_ids()]
    if len(gpu_ids) == 0:
        return [torch.device("cpu")]
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        mapped = []
        for gpu_id in gpu_ids:
            try:
                mapped.append(int(visible.index(gpu_id)))
            except ValueError:
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES set incorrectly. "
                    f"Got {os.environ['CUDA_VISIBLE_DEVICES']}, expected to include {gpu_id}. "
                )
    else:
        mapped = [int(gpu_id) for gpu_id in gpu_ids]
    return [torch.device(f"cuda:{i}") for i in mapped]


class ElasticWorker(BaseWorker):
    """ElasticWorker is a Ray actor that runs an elastic training job.

    This is different from "worker", which run old elastic agent.
    It skips the torch-run, and directly runs the training in process.
    """

    def setup(self):
        assert self.actor_info.spec.backend == "elastic"

        self._master_socket = None
        self._process_group_setup = False
        super().setup()

    def _setup_envs(self):
        """Setup environment variables for the worker."""

        # referenced ray.train.torch.config.py

        os.environ["NAME"] = self.actor_info.name
        os.environ["LOCAL_RANK"] = str(self.actor_info.local_rank)
        os.environ["RANK"] = str(self.actor_info.rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.actor_info.spec.per_group)
        os.environ["WORLD_SIZE"] = str(self.actor_info.spec.total)
        os.environ["NODE_RANK"] = str(self.actor_info.node_rank)

        # Setup device
        if self.job_info.accelerator_type == ACCELERATOR_TYPE.GPU:
            if self.actor_info.spec.rank_based_gpu_selection:
                device = torch.device(f"cuda:{self.actor_info.local_rank}")
            else:
                device = _get_ray_gpu_devices()[0]
        else:
            device = torch.device("cpu")
        os.environ["ACCELERATE_TORCH_DEVICE"] = str(device)
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.set_device(device)

    # region Rendezvous and process group setup

    def get_master_addr(self):
        """Get a master address for distributed training."""
        addr = ray.util.get_node_ip_address()

        # Release old master socket if exists
        self._release_master_socket()
        port, socket = find_free_port_from_env_and_bind(addr)
        self._master_socket = socket

        ret = f"tcp://{addr}:{port}"
        logger.info(f"Master address for distributed training: {ret}")
        return ret

    def _release_master_socket(self):
        if self._master_socket:
            self._master_socket.close()
            self._master_socket = None

    def setup_torch_process_group(
        self,
        master_addr: str,
        world_size: int,
        rank: int,
        only_envs: bool = False,
    ):
        """Setup the torch process group for distributed training."""
        assert self.actor_info.spec.backend == "elastic"
        backend = self.actor_info.spec.comm_backend
        if backend == "auto":
            backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout = (
            timedelta(seconds=self.actor_info.spec.comm_timeout_s)
            if self.actor_info.spec.comm_timeout_s is not None
            else None
        )
        logger.info(
            f"Setting up torch process group with backend={backend}, "
            f"world_rank={rank}, world_size={world_size}, by {master_addr}"
        )

        # Release the port to let torch use it.
        self._release_master_socket()

        # TODO backend specific setup
        if only_envs:
            addr, port = master_addr.split("://")[1].split(":")
            os.environ["MASTER_ADDR"] = addr
            os.environ["MASTER_PORT"] = str(port)
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            logger.info(
                f"Set MASTER_ADDR={addr}, MASTER_PORT={port} for torch distributed training."
            )
            return

        assert not self._process_group_setup, (
            "Torch process group is already set up. "
            "Call destroy_torch_process_group() before setting it up again."
        )

        with ElasticWorkerEvents.init_process_group():
            torch.distributed.init_process_group(
                backend=backend,
                init_method=master_addr,
                rank=rank,
                world_size=world_size,
                **(
                    {"timeout": timeout} if timeout else {}  # type:ignore
                ),  # old version torch<2.1 does not support timeout=None
            )
        self._process_group_setup = True

    def destroy_torch_process_group(self):
        """Destroy the torch process group."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Destroyed torch process group.")
        self._process_group_setup = False

        devices = _get_ray_gpu_devices()
        if torch.cuda.is_available():
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

    # endregion

    def get_ray_node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def run_network_check(self) -> float:
        """Run network check before starting the job."""
        try:
            from dlrover.python.unified.backend.elastic.worker.node_check import (
                run_comm_check,
            )

            logger.info(f"[{self.actor_info.name}] Running network check.")
            with ElasticWorkerEvents.comm_check() as span:
                res = run_comm_check()
                res = round(res, 3)  # round to 3 decimal places
                span.extra_args(result=res)
            logger.info(
                f"[{self.actor_info.name}] Network check finished, "
                f"result: {res:.3f} seconds."
            )
            return res
        except Exception as e:
            logger.error(
                f"[{self.actor_info.name}] Failed to run network check: {e}",
                exc_info=True,
            )
            return float("inf")  # return inf to indicate failure

    def start(self):
        "Noop, controlled by sub-master."

    def start_elastic_job(self):
        """Start the elastic worker. If already started, do nothing."""
        logger.info(f"Starting elastic worker {self.actor_info.name}.")
        super().start()

    def _on_execution_end(self, result: "ExecutionResult"):
        self.destroy_torch_process_group()
        super()._on_execution_end(result)
