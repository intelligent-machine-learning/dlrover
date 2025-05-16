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
import time
import typing
from abc import ABC, abstractmethod
from collections import Counter

from ray.exceptions import GetTimeoutError

from dlrover.python.common import env_utils
from dlrover.python.common.enums import ResourceType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import DLMasterConstant
from dlrover.python.unified.common.exception import ResourceError
from dlrover.python.unified.master.graph import (
    DLExecutionGraph,
    PlacementGroupAllocation,
)


class Placement(ABC):
    def __init__(self, execution_graph: DLExecutionGraph):
        self._graph = execution_graph

    @property
    def graph(self):
        return self._graph

    @classmethod
    def _get_start_bundle(cls, allocated_bundles, bundle_topology) -> int:
        if not allocated_bundles:
            return 0

        counter: typing.Counter[int] = Counter(allocated_bundles)
        max_bundle_index = max(counter.keys())
        for bundle_allocated in list(counter.items()):
            if bundle_allocated[1] >= bundle_topology[bundle_allocated[0]]:
                continue
            return int(bundle_allocated[0])

        return max_bundle_index + 1

    # create placement group by ray api
    def create_placement_group(self):
        start = time.time()

        try:
            self.graph.create_placement_group()
        except GetTimeoutError:
            logger.error("Got timeout when creating placement group.")
            raise ResourceError()

        logger.info(
            f"All placement group created used: {time.time() - start:.2f}s"
        )

    @classmethod
    @abstractmethod
    def get_placement_strategy(cls):
        """Return ray placement group strategy"""

    @abstractmethod
    def prepare_placement_group(self):
        """
        Construct placement group and bundles according to the group in
        described by rl-context.
        """

    @abstractmethod
    def allocate_placement_group(self):
        """
        Allocate the placement group prepared by 'prepare_placement_group' to
        workload actors.
        """


class SingleBundlePerNodePlacement(Placement):
    """
    Placement group: there is ony 1 placement group with 'STRICT_SPREAD'
        strategy in global.
    Bundle:
        bundle size = node's resource size
        bundle number = node's number
    """

    PG_NAME = "SINGLE_BUNDLE_PER_NODE"

    @classmethod
    def get_placement_strategy(cls):
        strategy_from_env = env_utils.get_env(DLMasterConstant.PG_STRATEGY_ENV)

        if not strategy_from_env:
            return "STRICT_SPREAD"

        logger.info(
            f"Override placement strategy from env: {strategy_from_env}."
        )
        return strategy_from_env

    def prepare_placement_group(self):
        strategy = self.get_placement_strategy()

        # define bundle unit resource
        device_per_node = self.graph.dl_context.trainer.device_per_node
        if self.graph.dl_context.trainer.device_type == ResourceType.CPU:
            bundle_unit_resource = {"CPU": device_per_node}
        else:
            bundle_unit_resource = {"GPU": device_per_node}

        # create bundle by nodes number
        bundles = []
        for i in range(self.graph.dl_context.trainer.node_number):
            bundles.append(bundle_unit_resource)

        pg_name = SingleBundlePerNodePlacement.PG_NAME
        logger.info(
            f"Prepare placement group {pg_name} with strategy: {strategy}, "
            f"bundles: {bundles}"
        )
        self.graph.add_placement_group(
            pg_name,
            PlacementGroupAllocation(pg_name, 0, strategy, bundles),
        )

    def allocate_placement_group(self):
        workload_group = self.graph.get_workload_group()
        pg = self.graph.get_placement_group(
            SingleBundlePerNodePlacement.PG_NAME
        )
        allocated_bundles = []
        bundle_topology = self.graph.get_bundle_topology()

        for group_desc_tuple in workload_group.groups:
            group_dict = group_desc_tuple[0]

            for role, role_group_size in group_dict.items():
                bundle_index = self._get_start_bundle(
                    allocated_bundles, bundle_topology
                )
                i = 0
                for vertex in self.graph.execution_vertices[role]:
                    vertex.update_pg_info(pg.pg_instance, bundle_index)
                    allocated_bundles.append(bundle_index)
                    pg.allocate(vertex.name, bundle_index)

                    i += 1
                    if i == role_group_size:
                        bundle_index += 1


class SingleGroupPerNodePlacement(Placement):
    """
    Placement group: there are multi placement groups, a placement group with
        'STRICT_PACK' strategy for each node.

    Bundle:
        bundle size = unit resource size(1 CPU/GPU)
        bundle number = node's resource size
    """

    PG_NAME = "SINGLE_GROUP_PER_NODE"

    @classmethod
    def get_placement_strategy(cls):
        strategy_from_env = env_utils.get_env(DLMasterConstant.PG_STRATEGY_ENV)

        if not strategy_from_env:
            return "STRICT_PACK"

        logger.info(
            f"Override placement strategy from env: {strategy_from_env}."
        )
        return strategy_from_env

    def prepare_placement_group(self):
        if self.graph.dl_context.trainer.device_type == ResourceType.CPU:
            bundle_unit_resource = {"CPU": 1}
        else:
            bundle_unit_resource = {"GPU": 1}

        strategy = self.get_placement_strategy()
        pg_num = self.graph.dl_context.trainer.node_number
        bundles = []
        for _ in range(self.graph.dl_context.trainer.device_per_node):
            bundles.append(bundle_unit_resource)

        for i in range(pg_num):
            pg_name = SingleGroupPerNodePlacement.PG_NAME + "_" + str(i)

            self.graph.add_placement_group(
                pg_name,
                PlacementGroupAllocation(pg_name, i, strategy, bundles),
            )

        logger.info(
            f"Prepare {pg_num} placement groups with strategy: {strategy}, "
            f"bundles: {bundles}"
        )

    def _get_all_pgs(self):
        return [
            value
            for _, value in sorted(
                self.graph.get_placement_group().items(),
                key=lambda item: item[0],
            )
        ]

    def allocate_placement_group(self):
        (
            collocation_group,
            common_group,
        ) = self.graph.dl_context.workload_group.split_groups_in_dict()
        assert len(collocation_group) >= 1

        pgs = self._get_all_pgs()

        # for collocation group 1st
        for name, group_tuple in collocation_group.items():
            pgs = [x for x in pgs if not x.is_full()]
            group_desc = group_tuple[0]

            for role, role_group_size in group_desc.items():
                for index, vertex in enumerate(
                    self.graph.execution_vertices[role]
                ):
                    pg = pgs[index // role_group_size]
                    bundle_index = index % role_group_size
                    logger.debug(
                        f"Allocate vertex {vertex.name} "
                        f"with pg: {pg.name}, "
                        f"bundle index: {bundle_index}"
                    )
                    vertex.update_pg_info(pg.pg_instance, bundle_index)
                    pg.allocate(vertex.name, bundle_index)

        # then common group
        for name, group_tuple in common_group.items():
            pgs = [x for x in pgs if not x.is_full()]
            group_desc = group_tuple[0]

            for role, role_group_size in group_desc.items():
                for index, vertex in enumerate(
                    self.graph.execution_vertices[role]
                ):
                    pg = pgs[index // role_group_size]
                    bundle_index = index % role_group_size
                    logger.debug(
                        f"Allocate vertex {vertex.name} "
                        f"with pg: {pg.name}, "
                        f"bundle index: {bundle_index}"
                    )
                    vertex.update_pg_info(pg.pg_instance, bundle_index)
                    pg.allocate(vertex.name, bundle_index)
