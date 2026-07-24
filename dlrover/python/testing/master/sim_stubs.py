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

"""No-op stub implementations of K8s-specific dependencies.

These stubs let DistributedJobMaster / DistributedJobManager run without a
real Kubernetes cluster.  They are used by sim_master_main.py which patches
the K8s factory functions before instantiating the master.

SimNodeWatcher.list() reads the live job_context so that the node monitor
thread always sees the correct node set and never marks agents as deleted.
"""

from typing import List

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeType,
)
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.master.watcher.base_watcher import NodeWatcher
from dlrover.python.scheduler.job import ElasticJob, JobArgs, NodeArgs

# Platform tag used for the simulation mode.  Any string not in
# [PlatformType.KUBERNETES, PlatformType.PY_KUBERNETES] works because
# DistributedJobMaster.__init__ only creates a K8s service for those two.
SIM_PLATFORM = "sim"


class SimElasticJob(ElasticJob):
    """No-op ElasticJob — never talks to Kubernetes."""

    def __init__(self):
        super().__init__("sim-job", "default")

    def get_node_service_addr(self, node_type, node_id):
        return "127.0.0.1:0"

    def get_node_name(self, node_type, node_id):
        return f"sim-{node_type}-{node_id}"


class SimNodeWatcher(NodeWatcher):
    """NodeWatcher that mirrors the in-process job_context.

    Reading directly from job_context ensures that
    DistributedJobManager._monitor_nodes never sees nodes as "missing from
    the cluster" and never generates spurious DELETED events for agents that
    are actually running.
    """

    def __init__(self):
        super().__init__("sim-job")

    def watch(self):
        """Return an empty iterator — we rely on heartbeats, not pod events."""
        return iter([])

    def list(self) -> List[Node]:
        """Return all nodes currently known to the master."""
        from dlrover.python.master.node.job_context import get_job_context

        nodes: List[Node] = []
        for type_nodes in get_job_context().dup_job_nodes().values():
            nodes.extend(type_nodes.values())
        return nodes


class SimScaler(Scaler):
    """No-op Scaler — never creates or deletes K8s pods."""

    def __init__(self):
        super().__init__("sim-job")

    def start(self):
        pass

    def scale(self, plan: ScalePlan, **kwargs):
        pass


class SimJobArgs(JobArgs):
    """AllReduce job args for simulation mode.

    Sets platform to ``SIM_PLATFORM`` so that DistributedJobMaster skips
    K8s service creation, and enables elastic scheduling so that
    DistributedJobManager is instantiated.
    """

    def __init__(
        self,
        num_workers: int = 4,
        max_relaunch_count: int = 3,
        job_name: str = "sim-job",
        namespace: str = "default",
    ):
        super().__init__(SIM_PLATFORM, namespace, job_name)
        self._num_workers = num_workers
        self._max_relaunch_count = max_relaunch_count

    def initilize(self):
        self.distribution_strategy = DistributionStrategy.ALLREDUCE
        self.enable_elastic_scheduling = True
        self.job_uuid = "sim-job-0001"
        worker_resource = NodeGroupResource(
            self._num_workers, NodeResource(0, 0)
        )
        self.node_args[NodeType.WORKER] = NodeArgs(
            worker_resource,
            auto_scale=False,
            restart_count=self._max_relaunch_count,
        )
