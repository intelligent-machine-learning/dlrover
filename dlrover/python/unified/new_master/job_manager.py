import time
from typing import TYPE_CHECKING, Dict, List

import ray

from dlrover.python.common.constants import NodeType
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.unified.common.constant import DLMasterConstant, DLWorkloadEnv
from dlrover.python.unified.common.enums import InternalRoleType
from dlrover.python.unified.master.graph import DLExecutionGraph
from dlrover.python.unified.master.scheduler import Scheduler

if TYPE_CHECKING:
    from dlrover.python.unified.new_master.core import Core

from dlrover.python.common.log import default_logger as logger


class JobManager:
    def __init__(self, core: "Core") -> None:
        self.graph = DLExecutionGraph(core.job_config.dl_context)
        self.scheduler = Scheduler.create(
            core.job_config.scheduling_strategy_type, self.graph
        )

        # state
        self.nodes = []

    def init_nodes(self):
        if self.nodes:
            logger.info("Nodes are already initialized.")
            return
        worker_nodes = []
        for elastic_vertex in self.graph.execution_vertices[
            InternalRoleType.ELASTIC.name
        ]:
            worker_nodes.append(
                Node(
                    node_type=NodeType.WORKER,
                    node_id=elastic_vertex.rank,
                    rank_index=elastic_vertex.rank,
                    name=elastic_vertex.name,
                    config_resource=NodeResource.resource_to_node_resource(
                        elastic_vertex.resource
                    ),
                    max_relaunch_count=elastic_vertex.extra_args.get(
                        "max_relaunch_count", 3
                    ),
                    service_addr=elastic_vertex.actor_handle,
                )
            )
        self.nodes.extend(worker_nodes)

    def create_nodes(self):
        self.scheduler.schedule()

    def precheck_nodes(self):
        pass

    def start_job(self):
        self._setup_workloads()

    def monitor_nodes(self):
        pass

    def stop_job(self):
        self.scheduler.cleanup()

    def _setup_workloads(self):
        """Sync operation."""
        logger.info("Start setup all workloads...")
        start = time.time() * 1000

        # envs for setup
        ports = self.graph.dl_context.trainer.torch_master_port
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
                    env_dict[DLWorkloadEnv.MASTER_ADDR] = runtime_info.host_ip
                    env_dict[DLWorkloadEnv.MASTER_PORT] = str(ports[i])
                    break
            env_dict_by_role[role] = env_dict
            i += 1

        timeout = max(
            DLMasterConstant.SETUP_TIMEOUT_MIN_SECS,
            len(self.graph.get_all_actor_handles())
            * DLMasterConstant.SETUP_TIMEOUT_PER_ACTOR_SECS,
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
                f"{len(not_ready)} workload actors setup timeout: {timeout}s."
            )

        end = time.time() * 1000 - start
        logger.info(f"Finish setup all workloads({len(ready)}), cost: {end:.2f}ms")
