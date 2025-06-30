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

from typing import List

from dlrover.python.common.constants import NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.watcher.base_watcher import NodeWatcher
from dlrover.python.scheduler.ray import RayClient
from dlrover.python.util.queue.queue import RayEventQueue


def check_actor_status(name):
    return "RUNNING"


def parse_event(msg):
    """
    parse event info from message
    """
    return msg


def parse_type(name):
    name = name.lower()
    node_type = None
    if NodeType.PS in name:
        node_type = NodeType.PS
    elif NodeType.EVALUATOR in name:
        node_type = NodeType.EVALUATOR
    elif NodeType.WORKER in name:
        node_type = NodeType.WORKER
    return node_type


def parse_index(name):
    """
    PsActor_1 split("_")[-1]
    TFSinkFunction-4|20 split("|").split("-")[-1]
    """
    node_type = parse_type(name)
    node_index = None
    if node_type == NodeType.PS:
        node_index = int(name.split("_")[-1])
    elif node_type == NodeType.EVALUATOR:
        node_index = 1
    elif node_type == NodeType.WORKER:
        node_index = int(name.split("|")[0].split("-")[-1])
    return node_index


def parse_type_id_from_actor_name(name):
    node_type = parse_type(name)
    node_index = parse_index(name)
    return node_type, node_index


class RayScalePlanWatcher:
    def __init__(self, job_name, namespace, job_uuid):
        self.job_name = job_name
        self.namespace = namespace
        self.job_uuid = job_uuid

    def watch(self):
        while True:
            yield None


class ActorWatcher(NodeWatcher):
    """ActorWatcher monitors all actors of a ray Job."""

    def __init__(self, job_name, namespace):
        self._job_name = job_name
        self._namespace = namespace
        self._ray_client = RayClient.singleton_instance(job_name, namespace)
        self.event_queue = RayEventQueue.singleton_instance()

    def watch(self):
        while True:
            i = self.event_queue.get()
            event = parse_event(i)
            logger.info(i)
            yield event

    def list(self) -> List[Node]:
        nodes: List[Node] = []
        for name, status in self._ray_client.list_actor():
            actor_type, actor_index = parse_type_id_from_actor_name(name)
            node = Node(
                node_type=actor_type,
                node_id=actor_index,
                name=actor_index,
                rank_index=actor_index,
                status=status,
                start_time=None,
            )
            nodes.append(node)
        return []
