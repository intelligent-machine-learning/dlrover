# Copyright 2022 The DLRover Authors. All rights reserved.
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

from typing import Dict

from dlrover.python.common.constants import DistributionStrategy, NodeType
from dlrover.python.common.resource import NodeGroupResource, NodeResource
from dlrover.python.master.node_watcher.base_watcher import Node


class JobResourceConfig(object):
    def __init__(self):
        self._group_resources: Dict[str, NodeGroupResource] = {}

    def add_node_group_resource(
        self, node_type, num, resource_config, priority
    ):
        self._group_resources[node_type] = NodeGroupResource(
            count=num,
            node_resource=NodeResource.resource_str_to_node_resource(
                resource_config
            ),
            priority=priority,
        )

    def get_node_group_resource(self, node_type):
        return self._group_resources.get(node_type, None)

    def _get_group_node_num(self, node_type):
        if node_type in self._group_resources:
            return self._group_resources[node_type].count
        return 0

    def get_node_types(self):
        return list(self._group_resources.keys())

    def update_node_group_resource(self, node_type, num, cpu, memory):
        self._group_resources.setdefault(
            node_type,
            NodeGroupResource(
                count=0,
                node_resource=NodeResource(0, 0),
                priority=None,
            ),
        )
        resource = self._group_resources[node_type]
        resource.count = num or resource.count
        resource.node_resource.cpu = cpu or resource.node_resource.cpu
        resource.node_resource.memory = memory or resource.node_resource.memory

    @property
    def worker_num(self):
        return self._get_group_node_num(NodeType.WORKER)

    @property
    def ps_num(self):
        return self._get_group_node_num(NodeType.PS)

    @property
    def evaluator_num(self):
        return self._get_group_node_num(NodeType.EVALUATOR)

    @property
    def tf_master_num(self):
        return self._get_group_node_num(NodeType.TF_MASTER)

    def init_job_node_meta(
        self,
        relaunch_on_worker_failure,
        service_create_fn,
    ):
        """
        job_resource: resource configuration of a job.
        relaunch_on_worker_failure: int, the number of relaunches.
        service_create_fn: a callable function to get the service address
            of a node.
        return: a dict with pod_type as key, and another dict as value.
                The other dict uses pod id as key, and PodInfo as value.
        """
        job_nodes: Dict[str, Dict[int, Node]] = {}
        for node_type in self.get_node_types():
            group_resource = self.get_node_group_resource(node_type)
            group_nodes: Dict[int, Node] = {}
            for i in range(group_resource.count):
                group_nodes[i] = Node(
                    node_type=node_type,
                    node_id=i,
                    max_relaunch_count=relaunch_on_worker_failure,
                    service_addr=service_create_fn(node_type, id),
                )
                group_nodes[i].update_resource_usage(
                    group_resource.node_resource.cpu,
                    group_resource.node_resource.memory,
                )
            job_nodes[node_type] = group_nodes
        return job_nodes


def set_critical_node(
    job_nodes: Dict[str, Dict[int, Node]],
    ps_is_critical=True,
    critical_worker_index={},
    ps_relaunch_max_num=0,
):
    """
    pod_info is a dict, where pod_info[type][id] is a PodInfo instance
    Set is_critical_pod values accordingly
    """
    if NodeType.PS in job_nodes:
        for node in job_nodes[NodeType.PS].values():
            node.critical = ps_is_critical
            if node.critical:
                node.max_relaunch_count = ps_relaunch_max_num
    if NodeType.WORKER in job_nodes:
        for i, node in job_nodes[NodeType.WORKER].items():
            if node.id not in critical_worker_index:
                continue
            node.critical = True
            node.max_relaunch_count = critical_worker_index[i]
    if NodeType.EVALUATOR in job_nodes:
        for node in job_nodes[NodeType.EVALUATOR].values():
            node.critical = True
    if NodeType.TF_MASTER in job_nodes:
        for node in job_nodes[NodeType.TF_MASTER].values():
            node.critical = True


def get_critical_worker_index(args):
    critical_worker_index = {}

    if args.critical_worker_index == "default":
        # for default, worker0 is critical if PS strategy with custom training
        if args.distribution_strategy == DistributionStrategy.PARAMETER_SERVER:
            critical_worker_index[0] = args.relaunch_on_worker_failure
    elif args.critical_worker_index == "all":
        for i in range(args.num_workers):
            critical_worker_index[i] = args.relaunch_on_worker_failure
    elif args.critical_worker_index != "none":
        for pod_relaunch_conf in args.critical_worker_index.split(","):
            # The conf is "pod_index:relaunch_times"
            pod_relaunch = pod_relaunch_conf.strip().split(":")
            critical_worker_index[int(pod_relaunch[0])] = int(pod_relaunch[1])

    return critical_worker_index
