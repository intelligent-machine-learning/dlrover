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

import asyncio
import time
from typing import Collection, List, OrderedDict, Sequence, Tuple

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.actor_base import ActorInfo
from dlrover.python.unified.util.actor_helper import (
    BatchInvokeResult,
    invoke_actor,
    invoke_actors,
    wait_batch_invoke,
)

from . import remote_call


def assert_sorted_by_rank(workers: List[ActorInfo]):
    """Assert that workers are sorted by rank."""
    for i, node in enumerate(workers):
        assert node.rank == i, (
            f"Node {node.name} has rank {node.rank}, expected {i}"
        )


async def group_by_node(
    workers: List[ActorInfo],
) -> List[List[ActorInfo]]:
    node_ids = await invoke_actors(
        remote_call.get_ray_node_id, [w.name for w in workers]
    )
    grouped: OrderedDict[str, List[ActorInfo]] = OrderedDict()
    for node, worker in zip(node_ids.results, workers):
        if node not in grouped:
            grouped[node] = []
        grouped[node].append(worker)
    return list(grouped.values())


class NodeCheckManager:
    """A class to manage node checks in a distributed system."""

    async def check_nodes(self, workers: List[ActorInfo]) -> List[float]:
        """Check Nodes before starting the job.

        Perform two round rendezvous to ensure all nodes are ready.
        1. Peer [1, 2] [3, 4] [5]
        2. Peer [Best, Worst] ...

        Return the minimum time from both rounds for each node.
        """

        assert_sorted_by_rank(workers)
        grouped = await group_by_node(workers)

        logger.info("Starting first round of grouping.")
        groups = self._grouping_round0(len(grouped))
        await self._setup_rendezvous_groups(
            workers,
            [
                [worker.rank for node in group for worker in grouped[node]]
                for group in groups
            ],
        )
        res_round0 = await self._perform_node_check(workers)
        await invoke_actors(
            remote_call.destroy_torch_process_group,
            workers,
        )
        logger.info("Completed round 0 of node checks.")
        # TODO handle fail cases
        logger.info(f"Node check results: {res_round0.as_dict()}")

        logger.info("Starting round 1 of node checks.")
        groups = self._grouping_round1(res_round0.results)  # TODO by node
        await self._setup_rendezvous_groups(workers, groups)
        res_round1 = await self._perform_node_check(workers)
        await invoke_actors(
            remote_call.destroy_torch_process_group,
            workers,
        )
        logger.info("Completed round 1 of node checks.")
        logger.info(f"Node check results: {res_round1.as_dict()}")

        # Return the minimum time from both rounds for each node.
        return [
            min(res) for res in zip(*[res_round0.results, res_round1.results])
        ]

    def find_abnormal_nodes(
        self, nodes: List[ActorInfo], times: List[float], threshold: float
    ) -> List[ActorInfo]:
        """Find nodes with times above a certain threshold."""
        return [node for node, t in zip(nodes, times) if t > threshold]

    def find_straggling_nodes(
        self, nodes: List[ActorInfo], times: List[float]
    ):
        """Find straggling nodes based on their times.
        Outside 3 standard deviations from the mean.
        """
        if not times or len(nodes) < 3:
            return []

        # Sort the times to find the mean and standard deviation of normal samples.
        sorted_times = sorted(times)
        normal_num = 2
        while normal_num < len(times):
            samples = sorted_times[:normal_num]
            mean_time = sum(samples) / len(samples)
            std_dev = (
                sum((t - mean_time) ** 2 for t in samples) / len(samples)
            ) ** 0.5
            threshold = mean_time + 3 * std_dev
            # Check if the next sample is above the threshold.
            if sorted_times[normal_num] > threshold:
                break
            normal_num += 1

        return self.find_abnormal_nodes(nodes, times, threshold)

    def _grouping_round0(self, size: int) -> List[Tuple[int, ...]]:
        """Group nodes just by their rank. like [1, 2], [3, 4], [5]."""
        groups: List[Tuple[int, ...]] = []
        for i in range(0, size, 2):
            if i + 1 < size:
                groups.append((i, i + 1))
            else:
                groups.append((i,))
        return groups

    def _grouping_round1(self, last_elapsed_time: List[float]):
        """Grouping nodes based on their round 0 times. like [Best, Worst]..."""
        sorted_nodes = sorted(
            range(len(last_elapsed_time)), key=lambda x: last_elapsed_time[x]
        )
        left, right = 0, len(sorted_nodes) - 1
        groups: List[Tuple[int, ...]] = []
        while left < right:
            groups.append((sorted_nodes[left], sorted_nodes[right]))
            left += 1
            right -= 1
        if left == right:
            groups.append((sorted_nodes[left],))  # odd one
        return groups

    async def _setup_rendezvous_group(
        self, group: List[ActorInfo], only_envs: bool = False
    ):
        master_addr = await invoke_actor(
            remote_call.get_master_addr, group[0].name
        )

        res = await wait_batch_invoke(
            invoke_actor(
                remote_call.setup_torch_process_group,
                node.name,
                master_addr=master_addr,
                world_size=len(group),
                rank=i,
                only_envs=only_envs,
            )
            for i, node in enumerate(group)
        )
        res.raise_for_errors()

    async def _setup_rendezvous_groups(
        self, nodes: List[ActorInfo], groups: Sequence[Collection[int]]
    ):
        logger.info(f"Groups for rendezvous: {groups}")
        start = time.time()
        await asyncio.gather(
            *[
                self._setup_rendezvous_group([nodes[rank] for rank in group])
                for group in groups
            ]
        )
        elapsed = time.time() - start
        logger.info(f"Rendezvous completed in {elapsed:.2f} seconds.")

    async def _perform_node_check(
        self, nodes: List[ActorInfo]
    ) -> BatchInvokeResult[float]:
        """Perform a node check."""
        res = await invoke_actors(
            remote_call.run_network_check, [node.name for node in nodes]
        )
        return res
