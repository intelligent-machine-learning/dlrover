import asyncio
from typing import List

import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import NodeInfo
from dlrover.python.unified.common.constant import (
    RAY_SINGLE_NODE_RELAUNCH_WAIT_TIME,
)
from dlrover.python.unified.controller.extension import Extension


@Extension.register_extension
class RelaunchSupport(Extension):
    async def relaunch_nodes_impl(self, nodes: List[NodeInfo]):
        logger.info(f"Relaunch nodes: {nodes}.")
        # TODO do_relaunch

        # wait for old nodes removed
        await asyncio.wait_for(
            self._wait_node_remove([node.id for node in nodes]),
            len(nodes) * RAY_SINGLE_NODE_RELAUNCH_WAIT_TIME,
        )

    def _running_ray_nodes(self):
        return [ray_node["NodeID"] for ray_node in ray.nodes()]

    async def _wait_node_remove(self, nodes: List[str], interval: float = 10):
        nodes = nodes.copy()
        while nodes:
            logger.info(f"Waiting for ray nodes removing: {nodes}")
            running = self._running_ray_nodes()
            nodes = [node for node in nodes if node in running]
            await asyncio.sleep(interval)


def test_node_relaunch():
    pass
