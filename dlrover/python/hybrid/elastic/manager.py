import asyncio
import threading
import time
from typing import List

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.hybrid.common.node_defines import NodeInfo, WorkerStage
from dlrover.python.hybrid.elastic.executor import ElasticExecutor
from dlrover.python.hybrid.util.actor_helper import invoke_actors_async
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
from dlrover.python.unified.common.enums import JobStage


def convert_to_node_state(state: WorkerStage):
    if state in ["INIT", "CHECKING"]:
        return NodeStatus.PENDING
    elif state in ["RUNNING"]:
        return NodeStatus.RUNNING
    elif state in ["FINISHED", "FAILED"]:
        return NodeStatus.FINISHED
    return NodeStatus.UNKNOWN


class ElasticManager:
    def __init__(self, nodes: List[NodeInfo]):
        self.job_name = "TODO"
        self.stage: JobStage = JobStage.INIT
        self.nodes: List[NodeInfo] = nodes

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
                max_relaunch_count=3,
            )
        self._old_context.update_job_nodes({NodeType.WORKER: group_nodes})

    async def start(self):
        # Initialize the elastic client here
        logger.info("Start job execution.")
        self.setup_workloads()
        await invoke_actors_async(
            [node.name for node in self.nodes], "start_elastic_job"
        )
        status = await invoke_actors_async(
            [node.name for node in self.nodes], "status"
        )
        if any(it != "RUNNING" for it in status):
            raise Exception("Some nodes failed to start the job.")
        self.stage = JobStage.RUNNING
        asyncio.create_task(self._monitor_nodes(), name="monitor_nodes")

    def setup_workloads(self):
        logger.info("Start setup all workloads...")
        start = time.time() * 1000
        # TODO setup rendezvous manager
        ready = []
        end = time.time() * 1000 - start
        logger.info(
            f"Finish setup all workloads({len(ready)}), cost: {end:.2f}ms"
        )

    async def _monitor_nodes(self):
        logger.info("Start monitoring nodes status...")
        while self.stage == JobStage.RUNNING:
            try:
                all_status: List[WorkerStage] = await invoke_actors_async(
                    [node.name for node in self.nodes], "status"
                )
                logger.debug(f"Node status results: {all_status}")
                with self._lock:
                    for node, status in zip(self.nodes, all_status):
                        old_node = self._old_context.job_node(
                            NodeType.WORKER, node.rank
                        )
                        assert old_node is not None, (
                            f"Node({node.rank}) not found in context"
                        )
                        status = convert_to_node_state(status)
                        old_node.update_status(status)
                        self._old_context.update_job_node(old_node)
                if all(it.is_terminal() for it in all_status):
                    self.stage = JobStage.FINISHED
                    logger.info("All nodes are finished.")
                    break
            except Exception as e:
                logger.warning(e)
                await asyncio.sleep(30)
            await asyncio.sleep(5)
