# Copyright 2023 The DLRover Authors. All rights reserved.
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

import datetime
import time
import unittest

from dlrover.python.common.constants import NetworkFailureReason
from dlrover.python.common.node import Node
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore
from dlrover.python.master.elastic_training.net_topology import (
    NodeTopologyMeta,
)
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
)
from dlrover.python.tests.test_utils import start_local_master


class MasterKVStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 0.5)

    def tearDown(self):
        self._master.stop()

    def test_kv_store_api(self):
        kv_store = MasterKVStore("dlrover/torch/test")
        key = "key0"
        kv_store.set(key, "1".encode())
        value = kv_store.get(key)
        self.assertEqual(int(value), 1)
        kv_store.add(key, 2)
        value = kv_store.get(key)
        self.assertEqual(int(value), 3)
        kv_store.wait([key])
        try:
            kv_store.wait(
                ["aa"], override_timeout=datetime.timedelta(seconds=0.01)
            )
        except Exception as e:
            self.assertIsInstance(e, LookupError)


class ElasticTrainingRendezvousManagerTest(unittest.TestCase):
    def test_max_nodes(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager.update_rdzv_params(3, 3, 60, 1)
        rdzv_round = rdzv_manager.get_rdzv_round()
        self.assertEqual(rdzv_round, 0)
        rdzv_manager._alive_nodes = [0, 1, 2]
        rdzv_manager.join_rendezvous(0, 0, 8)
        rdzv_manager.join_rendezvous(1, 1, 8)
        round, _, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 0)
        self.assertDictEqual(world, {})
        self.assertEqual(len(rdzv_manager._waiting_nodes), 2)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 0)
        rdzv_manager.join_rendezvous(2, 2, 8)
        self.assertDictEqual(
            rdzv_manager._node_rdzv_times, {0: 0.0, 1: 0.0, 2: 0.0}
        )
        round, _, world = rdzv_manager.get_comm_world(0)
        self.assertDictEqual(rdzv_manager._node_rdzv_times, {})
        self.assertEqual(round, 1)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 0)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 3)
        self.assertListEqual(list(world.keys()), [0, 1, 2])

    def test_min_nodes(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager.update_rdzv_params(2, 3, 0.1, 1)
        node_1 = Node("worker", 1)
        rdzv_manager.add_alive_node(node_1)
        node_0 = Node("worker", 0)
        rdzv_manager.add_alive_node(node_0)
        node_2 = Node("worker", 2)
        rdzv_manager.add_alive_node(node_2)
        rdzv_manager.join_rendezvous(0, 0, 8)
        rdzv_manager.join_rendezvous(1, 1, 8)
        rdzv_manager.remove_alive_node(node_2)
        self.assertEqual(len(rdzv_manager._alive_nodes), 2)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 2)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 0)
        time.sleep(0.2)
        round, _, world = rdzv_manager.get_comm_world(1)
        self.assertEqual(round, 1)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 0)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 2)
        self.assertListEqual(list(world.keys()), [0, 1])

    def test_min_nodes_with_unit(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        min_nodes = 8
        max_nodes = 12
        node_unit = 4
        rdzv_manager.update_rdzv_params(min_nodes, max_nodes, 0.1, node_unit)

        test_loop = 10
        for i in range(test_loop):
            node = Node("worker", i, name=f"worker-{i}")
            rdzv_manager.add_alive_node(node)
            rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(len(rdzv_manager._alive_nodes), test_loop)
        self.assertEqual(len(rdzv_manager._waiting_nodes), test_loop)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 0)
        time.sleep(0.2)
        round, _, world = rdzv_manager.get_comm_world(1)
        self.assertEqual(round, 1)
        self.assertEqual(
            len(rdzv_manager._waiting_nodes), test_loop - min_nodes
        )
        self.assertEqual(len(rdzv_manager._rdzv_nodes), min_nodes)
        self.assertListEqual(list(world.keys()), list(range(min_nodes)))
        round, _, world = rdzv_manager.get_comm_world(9)
        self.assertEqual(round, 1)
        self.assertFalse(9 in world)

        # Test the number of waiting nodes is less than the node unit.
        self.assertEqual(rdzv_manager.num_nodes_waiting(), 0)
        rdzv_manager.join_rendezvous(10, 10, 8)
        rdzv_manager.join_rendezvous(11, 11, 8)
        self.assertEqual(
            len(rdzv_manager._waiting_nodes), rdzv_manager.num_nodes_waiting()
        )
        self.assertEqual(
            rdzv_manager.num_nodes_waiting(), test_loop + 2 - min_nodes
        )
        node_10 = Node("worker", 10, name="worker-10")
        node_11 = Node("worker", 11, name="worker-11")

        # Test removing nodes from waiting nodes.
        rdzv_manager.add_alive_node(node_10)
        rdzv_manager.add_alive_node(node_11)
        rdzv_manager.remove_alive_node(node_10)
        rdzv_manager.remove_alive_node(node_11)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 2)

        # Test the number of waiting nodes is equal or
        # bigger than the node unit.
        for i in range(12, 16):
            rdzv_manager.join_rendezvous(i, i, 8)
        num = rdzv_manager.num_nodes_waiting()
        self.assertEqual(num, 6)
        rdzv_manager.clear_waiting_nodes()
        num = rdzv_manager.num_nodes_waiting()
        self.assertEqual(num, 0)

    def test_get_lacking_ranks(self):
        rdzv_manager = ElasticTrainingRendezvousManager()

        rdzv_manager._rdzv_params.min_nodes = 4
        rdzv_manager._waiting_nodes = {0: 0, 1: 1, 2: 2, 3: 3}
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [])

        rdzv_manager._rdzv_params.min_nodes = 5
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [4])

        rdzv_manager._rdzv_params.min_nodes = 3
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [])

        rdzv_manager._rdzv_params.min_nodes = 6
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [4, 5])

        rdzv_manager._rdzv_params.min_nodes = 4
        rdzv_manager._waiting_nodes = {}
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [0, 1, 2, 3])

        rdzv_manager._rdzv_params.min_nodes = 0
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [])


class NetworkCheckRendezvousManagerTest(unittest.TestCase):
    def test_network_check_rdzv(self):
        rdzv_manager = NetworkCheckRendezvousManager()
        rdzv_manager.update_rdzv_params(4, 4, 60, 1)
        rdzv_manager._alive_nodes = [0, 1, 2, 3]
        for i in range(4):
            round = rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(round, 0)
        round, group, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 1)
        self.assertEqual(group, 0)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 0)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 4)
        self.assertListEqual(list(world.keys()), [0, 1])
        self.assertEqual(group, 0)
        round, group, world = rdzv_manager.get_comm_world(2)
        self.assertEqual(round, 1)
        self.assertListEqual(list(world.keys()), [2, 3])
        self.assertEqual(group, 1)
        for i in range(3):
            rdzv_manager.report_network_check_result(i, True, 0.0)
        rdzv_manager.report_network_check_result(3, False, 0.0)

        for i in range(4):
            round = rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(round, 1)
        round, group, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 2)
        self.assertListEqual(list(world.keys()), [0, 3])
        round, group, world = rdzv_manager.get_comm_world(1)
        self.assertEqual(round, 2)
        self.assertListEqual(list(world.keys()), [1, 2])
        self.assertEqual(group, 1)
        success, _ = rdzv_manager.check_fault_node()
        self.assertFalse(success)

        for i in range(4):
            round = rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(round, 2)
        round, group, world = rdzv_manager.get_comm_world(3)
        self.assertEqual(round, 3)
        self.assertListEqual(list(world.keys()), [2, 3])
        _, reason = rdzv_manager.check_fault_node()
        self.assertEqual(reason, NetworkFailureReason.WAITING_NODE)
        for i in range(3):
            rdzv_manager.report_network_check_result(i, True, 0.0)
        rdzv_manager.report_network_check_result(3, True, 0.0)
        nodes, _ = rdzv_manager.check_fault_node()
        self.assertListEqual(nodes, [])

    def test_network_check_rdzv_with_single_node(self):
        rdzv_manager = NetworkCheckRendezvousManager()
        rdzv_manager.update_rdzv_params(1, 1, 60, 1)
        rdzv_manager._alive_nodes = [0]
        round = rdzv_manager.join_rendezvous(0, 0, 8)
        self.assertEqual(round, 0)
        round, _, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 1)
        self.assertListEqual(list(world.keys()), [0])
        rdzv_manager.report_network_check_result(0, True, 0.0)
        nodes, _ = rdzv_manager.check_fault_node()
        self.assertListEqual(nodes, [])
        rdzv_manager._clear_check_status()
        self.assertDictEqual(rdzv_manager._node_times, {})
        self.assertDictEqual(rdzv_manager._node_status, {})

    def test_network_check_straggler_even_nodes(self):
        rdzv_manager = NetworkCheckRendezvousManager()
        rdzv_manager.update_rdzv_params(6, 6, 60, 1)
        rdzv_manager._alive_nodes = [0, 1, 2, 3, 4, 5]
        for i in range(6):
            round = rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(round, 0)
        round, group, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 1)
        self.assertEqual(group, 0)
        round, group, world = rdzv_manager.get_comm_world(2)
        self.assertEqual(round, 1)
        self.assertListEqual(list(world.keys()), [2, 3])
        self.assertEqual(group, 1)
        for i in range(4):
            rdzv_manager.report_network_check_result(i, True, 5.0)
        for i in range(4, 6):
            rdzv_manager.report_network_check_result(i, True, 15.0)
        stragglers, _ = rdzv_manager.get_straggler()
        self.assertListEqual(stragglers, [4, 5])

        for i in range(6):
            round = rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(round, 1)
        round, group, world = rdzv_manager.get_comm_world(5)
        self.assertEqual(round, 2)
        self.assertListEqual(list(world.keys()), [0, 5])

        for i in [1, 2, 3, 4]:
            rdzv_manager.report_network_check_result(i, True, 5.0)
        for i in [0, 5]:
            rdzv_manager.report_network_check_result(i, True, 15.0)
        stragglers, _ = rdzv_manager.get_straggler()
        self.assertListEqual(stragglers, [5])

    def test_network_check_straggler_old_nodes(self):
        rdzv_manager = NetworkCheckRendezvousManager()
        rdzv_manager.update_rdzv_params(5, 5, 60, 1)
        rdzv_manager._alive_nodes = [0, 1, 2, 3, 4]
        for i in range(5):
            round = rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(round, 0)
        round, group, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 1)
        self.assertEqual(group, 0)
        round, group, world = rdzv_manager.get_comm_world(2)
        self.assertEqual(round, 1)
        self.assertListEqual(list(world.keys()), [2, 3, 4])
        self.assertEqual(group, 1)
        for i in range(2):
            rdzv_manager.report_network_check_result(i, True, 15.0)
        for i in range(2, 5):
            rdzv_manager.report_network_check_result(i, True, 5.0)
        stragglers, _ = rdzv_manager.get_straggler()
        self.assertListEqual(stragglers, [0, 1])

        for i in range(5):
            round = rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(round, 1)
        round, group, world = rdzv_manager.get_comm_world(1)
        self.assertEqual(round, 2)
        self.assertListEqual(list(world.keys()), [2, 1])

        for i in [1, 2]:
            rdzv_manager.report_network_check_result(i, True, 15.0)
        for i in [0, 3, 4]:
            rdzv_manager.report_network_check_result(i, True, 5.0)
        stragglers, _ = rdzv_manager.get_straggler()
        self.assertListEqual(stragglers, [1])

    def test_sync_ckpt_nodes(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager._latest_rdzv_nodes = [0, 1]
        success = rdzv_manager.sync_ckpt_nodes(0, 100)
        self.assertFalse(success)
        success = rdzv_manager.sync_ckpt_nodes(1, 100)
        self.assertTrue(success)
        success = rdzv_manager.sync_ckpt_nodes(1, 90)
        self.assertFalse(success)

    def test_map_node_rank_to_id(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager._rdzv_nodes[0] = NodeTopologyMeta(
            node_id=1,
            node_rank=0,
            process_num=8,
        )
        rank_d = {0: True}
        id_d = rdzv_manager._map_node_rank_to_id(rank_d)
        self.assertDictEqual(id_d, {1: True})

    def test_when_node_not_init(self):
        rdzv_manager = NetworkCheckRendezvousManager()
        self.assertTrue(not rdzv_manager._rdzv_nodes)

        rdzv_manager.check_fault_node()
