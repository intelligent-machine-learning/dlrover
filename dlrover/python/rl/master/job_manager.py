# Copyright 2025 The EasyDL Authors. All rights reserved.
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
import time

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.rl.common.constant import RLMasterConstant, RLWorkloadEnv
from dlrover.python.rl.common.enums import SchedulingStrategyType
from dlrover.python.rl.common.job_context import get_job_context
from dlrover.python.rl.master.executor import Executor
from dlrover.python.rl.master.graph import RLExecutionGraph
from dlrover.python.rl.master.scheduler import (
    GroupOrderedScheduler,
    SimpleScheduler,
)


class JobManager(object):
    """
    Core job life cycle management.
    """

    def __init__(self):
        self._job_ctx = get_job_context()

        self._execution_graph = RLExecutionGraph(self._job_ctx.rl_context)
        self._job_ctx.set_execution_graph(self._execution_graph)

        self._scheduler = self._get_scheduler()
        self._executor = Executor(self._execution_graph)

    @property
    def graph(self):
        return self._execution_graph

    def _get_scheduling_type_from_context(self):
        return self._job_ctx.job_config.scheduling_strategy_type

    def _get_scheduler(self):
        strategy_type = self._get_scheduling_type_from_context()
        if strategy_type == SchedulingStrategyType.SIMPLE:
            logger.info("Use simple strategy for scheduling by specification.")
            return SimpleScheduler(self._execution_graph)
        elif strategy_type == SchedulingStrategyType.GROUP:
            if self._job_ctx.rl_context.has_workload_group():
                logger.info(
                    "Use group strategy for scheduling by specification."
                )
                return GroupOrderedScheduler(self._execution_graph)
            else:
                logger.info(
                    "Downgrade to simple strategy for scheduling because "
                    "workload group description is empty in rl-context."
                )
                return SimpleScheduler(self._execution_graph)
        else:
            # for auto type:
            # use group strategy if exits group in context,
            # or use simple strategy
            if self._job_ctx.rl_context.has_workload_group():
                logger.info("Use group strategy for scheduling by auto.")
                return GroupOrderedScheduler(self._execution_graph)
            else:
                logger.info("Use simple strategy for scheduling by auto.")
                return SimpleScheduler(self._execution_graph)

    def start_job(self):
        logger.info("Start job execution.")

        # create all workloads
        self.create_workloads()

        # setup all workloads
        self.setup_workloads()

        # execute trainer
        self.execute()

    def stop_job(self):
        self._reset()

        # destroy all workloads
        self.destroy_workloads()

    def _reset(self):
        pass

    def create_workloads(self):
        """Sync operation."""
        logger.info("Start creating all workloads...")

        self._scheduler.schedule()

    def _check_runtime_info(self):
        if any(
            create_vertex.create_time == 0
            for create_vertex in self.graph.get_all_vertices()
        ):
            logger.info(
                "Still waiting actor creation callback for "
                "updating runtime info..."
            )
            return False
        return True

    def setup_workloads(self):
        """Sync operation."""
        logger.info("Start setup all workloads...")
        start = time.time() * 1000

        # envs for setup
        ports = self.graph.rl_context.trainer.torch_master_port
        env_dict_by_role = {}
        i = 0
        for role, vertices in self.graph.execution_vertices.items():
            env_dict = {}
            # master addr and port
            for vertex in vertices:
                if vertex.rank == 0:
                    runtime_info = ray.get(
                        vertex.actor_handle.get_runtime_info.remote()
                    )
                    env_dict[RLWorkloadEnv.MASTER_ADDR] = runtime_info.host_ip
                    env_dict[RLWorkloadEnv.MASTER_PORT] = str(ports[i])
                    break
            env_dict_by_role[role] = env_dict
            i += 1

        timeout = max(
            RLMasterConstant.SETUP_TIMEOUT_MIN_SECS,
            len(self.graph.get_all_actor_handles())
            * RLMasterConstant.SETUP_TIMEOUT_PER_ACTOR_SECS,
        )

        setup_refs = [
            vertex.actor_handle.setup.remote(env_dict_by_role[vertex.role])
            for vertex in self.graph.get_all_vertices()
        ]
        ready, not_ready = ray.wait(
            setup_refs,
            num_returns=len(setup_refs),
            timeout=timeout,
        )
        if len(not_ready) > 0:
            raise TimeoutError(
                f"{len(not_ready)} workload actors "
                f"setup timeout: {timeout}s."
            )

        end = time.time() * 1000 - start
        logger.info(
            f"Finish setup all workloads({len(ready)}), cost: {end:.2f}ms"
        )

    def execute(self):
        self._executor.execute()

    def destroy_workloads(self):
        """Sync operation."""

        logger.info("Start destroying all workloads...")
        self._scheduler.cleanup()

    def is_job_finished(self):
        return self._executor.is_trainer_finished()

    def is_trainer_error(self):
        return self._executor.is_trainer_error()

    def get_trainer_error(self):
        return self._executor.get_trainer_error()
