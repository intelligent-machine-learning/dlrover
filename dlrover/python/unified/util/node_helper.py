#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
from typing import List

import ray

from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.util.decorators import log_execution
from dlrover.python.common.log import default_logger as logger


@log_execution("wait_ray_node_relaunch")
async def wait_ray_node_relaunching(
    nodes: List[str], total: int, interval: float = 5
):
    """
    Wait for Ray nodes to be relaunched.

    Args:
        nodes: List of node IDs to wait for removal.
        total: Total number of nodes.
        interval: Waiting interval in seconds.
    """

    if not nodes:
        return

    nodes = nodes.copy()
    relaunched_size = len(nodes)

    # wait node removing
    while nodes:
        running = set(
            ray_node["NodeID"] for ray_node in ray.nodes() if ray_node["Alive"]
        )
        nodes = [node for node in nodes if node in running]
        if nodes:
            logger.info(f"Waiting for ray nodes removing: {nodes}")
            await asyncio.sleep(interval)

    logger.info(f"Nodes already removed for relaunching: {nodes}")

    # wait new node ready
    while True:
        running_size = len(
            set(
                ray_node["NodeID"]
                for ray_node in ray.nodes()
                if ray_node["Alive"]
            )
        )
        if running_size < total:
            logger.info(
                f"Waiting for relaunched nodes to be ready, current: {running_size}, expected: {total}"
            )
            await asyncio.sleep(interval)
        else:
            logger.info(
                f"All relaunched nodes(size:{relaunched_size}) are ready."
            )
            break


def get_node_group(target_node: NodeInfo, group_label: str) -> List[NodeInfo]:
    """
    Get node groups for a given node.

    Args:
        target_node: Node to get node group label.
        group_label: Node group label key.

    Returns: Node info(with id only) in list.
    """

    ray_nodes = ray.nodes()
    group_label_val = ""

    for node in ray_nodes:
        if node["NodeID"] == target_node.id:
            group_label_val = node["Labels"].get(group_label, "")
            break

    if group_label_val:
        node_group = []
        for node in ray_nodes:
            if node["Labels"].get(group_label, "") == group_label_val:
                node_group.append(NodeInfo(id=node["NodeID"]))
        return node_group

    # return current node if no group label value
    return [target_node]
