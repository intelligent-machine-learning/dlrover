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

import importlib
import os
from contextlib import contextmanager
from datetime import timedelta
from threading import Thread
from typing import Callable

import ray
import ray.train.torch as ray_train
import torch

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.workload_base import ActorBase, WorkerStage
from dlrover.python.unified.util.os_util import get_free_port


@ray.remote
class ElasticWorker(ActorBase):
    """ElasticWorker is a Ray actor that runs an elastic training job.

    This is different from "worker", which run old elastic agent.
    It skips the torch-run, and directly runs the training in process.
    """

    def _setup(self):
        assert self.node_info.spec.backend == "elastic"

        self._process_group_setup = False

        self._setup_envs()
        # RayMasterClient.register_master_actor(f"{self.node_info.role}-master")

    def _setup_envs(self):
        """Setup environment variables for the worker."""

        # referenced ray.train.torch.config.py

        os.environ["LOCAL_RANK"] = str(self.node_info.local_rank)
        os.environ["RANK"] = str(self.node_info.rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(self.node_info.spec.per_group)
        os.environ["WORLD_SIZE"] = str(self.node_info.spec.total)
        os.environ["NODE_RANK"] = str(self.node_info.node_rank)

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
        """Get a master address for distributed training."""
        addr = ray.util.get_node_ip_address()
        port = get_free_port()

        return f"tcp://{addr}:{port}"

    def setup_torch_process_group(
        self, master_addr: str, world_size: int, rank: int
    ):
        """Setup the torch process group for distributed training."""
        assert self.node_info.spec.backend == "elastic"
        backend = self.node_info.spec.comm_backend
        timeout = timedelta(seconds=self.node_info.spec.comm_timeout_s)
        logger.info(
            f"Setting up torch process group with backend={backend}, "
            f"world_rank={rank}, world_size={world_size}, by {master_addr}"
        )
        # TODO backend specific setup

        assert not self._process_group_setup, (
            "Torch process group is already set up. "
            "Call destroy_torch_process_group() before setting it up again."
        )

        torch.distributed.init_process_group(
            backend=backend,
            init_method=master_addr,
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )
        self._process_group_setup = True

    def destroy_torch_process_group(self):
        """Destroy the torch process group."""
        devices = ray_train.get_devices()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Destroyed torch process group.")
        self._process_group_setup = False

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

            logger.info(f"[{self.node_info.name}] Running network check.")
            res = run_comm_check()
            res = round(res, 3)  # round to 3 decimal places
            logger.info(
                f"[{self.node_info.name}] Network check finished, "
                f"result: {res:.3f} seconds."
            )
            return res
        except Exception as e:
            logger.error(
                f"[{self.node_info.name}] Failed to run network check: {e}",
                exc_info=True,
            )
            return float("inf")  # return inf to indicate failure

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

    def _load_user_func(self, entry_point: str) -> Callable[..., object]:
        """Load the user function from the entry point specified in the workload spec."""

        if not entry_point or "::" not in entry_point:
            raise ValueError(
                "Entry point is not specified in the workload spec. "
                "It should be in the format 'module::function'."
            )
        module_name, func = entry_point.split("::", 1)
        logger.info(
            f"Running elastic job with entry point: {module_name}.{func}. "
        )

        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func)
            if not callable(func):
                raise ValueError(
                    f"Entry point {entry_point} is not a callable function."
                )
            return func
        except ImportError:
            logger.error(
                f"Failed to import module {module_name} for elastic job.",
            )
            raise
        except AttributeError:
            logger.error(
                f"Failed to get function {func} from module {module_name} for "
                f"elastic job.",
            )
            raise

    def _run_agent(self):
        """Run the elastic agent."""
        assert self.node_info.spec.backend == "elastic"

        run_user_func = self._load_user_func(self.node_info.spec.entry_point)
        run_user_func()

        logger.info("Done elastic training.")
