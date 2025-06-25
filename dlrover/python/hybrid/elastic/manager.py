import threading
import time
from typing import List

from dlrover.python.common.constants import NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.hybrid.elastic.executor import ElasticExecutor
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
)
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.job_context import (
    get_job_context as get_elastic_context,
)
from dlrover.python.master.watcher.ray_watcher import ActorWatcher
from dlrover.python.unified.common.constant import (
    DLMasterConstant,
    DLWorkloadEnv,
)
from dlrover.python.unified.common.enums import JobStage
from dlrover.python.unified.master.graph import DLExecutionVertex


class ElasticManager:
    def __init__(self, config):
        self.config = config

        self.job_name = "TODO"
        self.stage: JobStage = JobStage.INIT
        self.nodes: List[DLExecutionVertex] = []

        self.perf_monitor = PerfMonitor()
        self.diagnosis = DiagnosisMaster()
        self.rdzv_manager = ElasticTrainingRendezvousManager()
        self.node_check_manager = NetworkCheckRendezvousManager()
        self.executor = ElasticExecutor(self.nodes)
        self.node_watcher = ActorWatcher(self.job_name, "default")

        # This is old singleton context, used for compatibility
        self._lock = threading.Lock()
        self._old_context = get_elastic_context()

    def _prepare(self):
        self._init_old_context()

    def _init_old_context(self):
        group_nodes = {}
        for elastic_vertex in self.nodes:
            group_nodes[elastic_vertex.rank] = Node(
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
        self._old_context.update_job_nodes({NodeType.WORKER: group_nodes})

    def start(self):
        # Initialize the elastic client here
        logger.info("Start job execution.")
        self.setup_workloads()
        self.executor.execute()
        self._thread = threading.Thread(
            target=self._monitor_nodes, name="node_monitor", daemon=True
        )
        self._thread.start()

    def setup_workloads(self):
        logger.info("Start setup all workloads...")

        start = time.time() * 1000
        ports = self.graph.dl_context.trainer.torch_master_port
        env_dict_by_role = {}
        i = 0
        for role, vertices in self.graph.execution_vertices.items():
            env_dict = {}
            for vertex in vertices:
                if vertex.rank == 0:
                    import ray

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
        import ray

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
        logger.info(
            f"Finish setup all workloads({len(ready)}), cost: {end:.2f}ms"
        )

    def _monitor_nodes(self):
        logger.info("Start monitoring nodes status...")
        while True:
            try:
                list_nodes = self.node_watcher.list()
                for list_node in list_nodes:
                    node_id = list_node.id
                    with self._lock:
                        current_node = self._old_context.job_node(
                            NodeType.WORKER, node_id
                        )
                        current_node.status = list_node.status
                        self._old_context.update_job_node(current_node)
            except Exception as e:
                logger.warning(e)
                time.sleep(30)
            time.sleep(5)
