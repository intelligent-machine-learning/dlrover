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
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
)


class MasterKVStoreTest(unittest.TestCase):
    def test_kv_store_api(self):
        kv_store = MasterKVStore("dlrover/torch/test")
        key = "key0"
        kv_store.set(key, "1")
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
        rdzv_manager.update_rdzv_params(3, 3, 60)
        rdzv_manager._alive_nodes = [0, 1, 2]
        rdzv_manager.join_rendezvous(0, 8)
        rdzv_manager.join_rendezvous(1, 8)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 2)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 0)
        rdzv_manager.join_rendezvous(2, 8)
        _, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 0)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 3)
        self.assertDictEqual(world, {0: 8, 1: 8, 2: 8})

    def test_min_nodes(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager.update_rdzv_params(2, 3, 0.1)
        node_1 = Node("worker", 1)
        rdzv_manager.add_alive_node(node_1)
        node_0 = Node("worker", 0)
        rdzv_manager.add_alive_node(node_0)
        node_2 = Node("worker", 2)
        rdzv_manager.add_alive_node(node_2)
        rdzv_manager.join_rendezvous(0, 8)
        rdzv_manager.join_rendezvous(1, 8)
        rdzv_manager.remove_alive_node(node_2)
        self.assertEqual(len(rdzv_manager._alive_nodes), 2)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 2)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 0)
        time.sleep(0.2)
        _, world = rdzv_manager.get_comm_world(1)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 0)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 2)
        self.assertDictEqual(world, {0: 8, 1: 8})


class NcclCheckRendezvousManagerTest(unittest.TestCase):
    def test_network_check_rdzv(self):
        rdzv_manager = NetworkCheckRendezvousManager()
        rdzv_manager.update_rdzv_params(4, 4, 60)
        rdzv_manager._alive_nodes = [0, 1, 2, 3]
        for i in range(4):
            round = rdzv_manager.join_rendezvous(i, 8)
        self.assertEqual(round, 0)
        group, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(group, 0)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 0)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 4)
        self.assertDictEqual(world, {0: 8, 1: 8, 2: 8, 3: 8})
        for i in range(4):
            round = rdzv_manager.join_rendezvous(i, 8)
        self.assertEqual(round, 1)
        group, world = rdzv_manager.get_comm_world(0)
        self.assertDictEqual(world, {0: 8, 1: 8})
        group, world = rdzv_manager.get_comm_world(1)
        self.assertEqual(group, 0)
        self.assertDictEqual(world, {0: 8, 1: 8})
        group, world = rdzv_manager.get_comm_world(2)
        self.assertDictEqual(world, {2: 8, 3: 8})
        self.assertEqual(group, 1)
        for i in range(3):
            rdzv_manager.report_network_check_result(i, True)
        rdzv_manager.report_network_check_result(3, False)
        success, _ = rdzv_manager.network_check_success()
        self.assertFalse(success)
        for i in range(4):
            round = rdzv_manager.join_rendezvous(i, 8)
        self.assertEqual(round, 2)
        group, world = rdzv_manager.get_comm_world(3)
        self.assertDictEqual(world, {0: 8, 3: 8})
        _, reason = rdzv_manager.network_check_success()
        self.assertEqual(reason, NetworkFailureReason.WAITING_NODE)
        for i in range(3):
            rdzv_manager.report_network_check_result(i, True)
        rdzv_manager.report_network_check_result(3, True)
        success, _ = rdzv_manager.network_check_success()
        self.assertTrue(success)
