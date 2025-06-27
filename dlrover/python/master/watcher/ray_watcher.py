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

from ray.util.state import list_actors

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.watcher.base_watcher import NodeWatcher
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


def parse_from_actor_name(name):
    split_name = name.split("_")
    if len(split_name) == 1:
        return name
    elif len(split_name) == 2:
        # role_${index}
        return split_name[0], split_name[1]
    elif len(split_name) == 3:
        # role_${size}-${index}_${local_size}-${local_index}
        world_rank = split_name[1].split("-")
        local_rank = split_name[2].split("-")

        return (
            split_name[0],
            world_rank[0],
            world_rank[1],
            local_rank[0],
            local_rank[1],
        )
    else:
        return name


def parse_from_actor_state(state):
    """
    Ref:
    https://docs.ray.io/en/latest/ray-observability/reference/doc/
    ray.util.state.common.ActorState.html#ray.util.state.common.ActorState
    """

    if state in ["DEPENDENCIES_UNREADY", "PENDING_CREATION"]:
        return NodeStatus.PENDING
    elif state in ["ALIVE", "RESTARTING"]:
        return NodeStatus.RUNNING
    elif state in ["DEAD"]:
        return NodeStatus.DELETED
    return NodeStatus.UNKNOWN


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

    def __init__(self, job_uuid, namespace):
        super().__init__(job_uuid)
        self._namespace = namespace
        self.event_queue = RayEventQueue.singleton_instance()

    def watch(self):
        while True:
            i = self.event_queue.get()
            event = parse_event(i)
            logger.info(i)
            yield event

    def list(self, actor_class=None) -> List[Node]:
        if not actor_class:
            filters = [("class_name", "=", "ElasticWorkload")]
        else:
            filters = [("class_name", "=", actor_class)]

        nodes: List[Node] = []

        actor_states = list_actors(filters=filters, limit=1000)
        for actor_state in actor_states:
            actor_name = actor_state.name
            _, actor_size, actor_index, _, _ = parse_from_actor_name(
                actor_name
            )
            node = Node(
                node_type=NodeType.WORKER,
                node_id=actor_index,
                name=actor_name,
                rank_index=actor_index,
                status=parse_from_actor_state(actor_state.state),
            )
            nodes.append(node)
        return nodes
