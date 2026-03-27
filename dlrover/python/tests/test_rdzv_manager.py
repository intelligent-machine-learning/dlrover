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
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import dlrover.python.util.store_util as store_util
from dlrover.python.common.global_context import Context
from dlrover.python.common.constants import (
    GroupNodeCheckPhase,
    NetworkFailureReason,
    NodeEventType,
)
from dlrover.python.common.node import Node
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore
from dlrover.python.master.elastic_training.kv_store_service import (
    KVStoreService,
)
from dlrover.python.master.elastic_training.net_topology import (
    NodeTopologyMeta,
)
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    GroupNodeNetworkCheckRendezvousManager,
    NetworkCheckRendezvousManager,
    UcpRdzvManager,
    create_training_rdzv_manager,
)
from dlrover.python.tests.test_utils import start_local_master


class MasterKVStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 0.5)

    def tearDown(self):
        self._master.stop()

    def test_kv_store_service(self):
        kv_store = KVStoreService()
        kv_store.set("key0", 1)
        self.assertEqual(kv_store.get("key0"), 1)
        self.assertEqual(kv_store.get("key1"), b"")
        kv_store.add("key1", 2)
        self.assertEqual(kv_store.get("key1"), 2)
        kv_store.add("key1", 3)
        self.assertEqual(kv_store.get("key1"), 5)
        kv_store.clear()
        self.assertEqual(kv_store.get("key0"), b"")
        self.assertEqual(kv_store.get("key1"), b"")

    def test_kv_store_api(self):
        kv_store = MasterKVStore("dlrover/torch/test")
        kv_store.set_timeout(datetime.timedelta(seconds=0.5))

        key = "key0"
        kv_store.set(key, 1)
        self.assertEqual(kv_store.get(key), 1)

        kv_store.set(key, "1".encode())
        value = kv_store.get(key)
        self.assertEqual(value.decode(), "1")
        self.assertEqual(value, b"1")
        self.assertEqual(int(value), 1)

        kv_store.set(key, "abc".encode())
        value = kv_store.get(key)
        self.assertEqual(value, b"abc")
        self.assertEqual(value.decode(), "abc")

        with self.assertRaises(LookupError):
            kv_store.get("dummy")

        key = "key1"
        kv_store.add(key, 2)
        self.assertEqual(kv_store.get(key), 2)
        kv_store.add(key, 3)
        self.assertEqual(kv_store.get(key), 5)

        kv_store.wait([key])
        with self.assertRaises(LookupError):
            kv_store.wait(
                ["aa"], override_timeout=datetime.timedelta(seconds=0.5)
            )

        self.assertEqual(kv_store.check([key]), True)
        self.assertEqual(kv_store.check("foo"), False)

        kv_store.add("key2", 100)
        kv_store.add("key3", 200)
        self.assertEqual(kv_store.multi_get(["key2", "key3"]), [100, 200])

        with self.assertRaises(LookupError):
            kv_store.multi_get(["key2", "key3", "key4"])

        kv_store.multi_set(["foo", "bar"], ["foo1", "bar1"])
        self.assertEqual(kv_store.multi_get(["foo", "bar"]), ["foo1", "bar1"])
        self.assertEqual(kv_store.get("foo"), "foo1")
        self.assertEqual(kv_store.get("bar"), "bar1")

        with self.assertRaises(IndexError):
            kv_store.multi_set(["foo", "bar"], ["foo1"])

    def test_kv_store_timeout(self):
        kv_store = MasterKVStore("dlrover/torch/test")
        key1 = "alpha"
        key2 = "beta"
        key3 = "omega"
        kv_store.set(key1, "1".encode())
        kv_store.set(key2, "2".encode())
        kv_store.wait([key1, key2])
        self.assertEqual("1".encode(), kv_store.get(key1))
        self.assertEqual("2".encode(), kv_store.get(key2))

        kv_store.set_timeout(datetime.timedelta(seconds=1))
        try:
            kv_store.wait([key1, key2, key3])
        except Exception as e:
            self.assertIsInstance(e, LookupError)

        try:
            kv_store.get(key3)
        except Exception as e:
            self.assertIsInstance(e, LookupError)

    def test_store_util(self):
        store = MasterKVStore("dlrover/torch/test1")
        store.set_timeout(datetime.timedelta(seconds=1))
        key_prefix = "test"

        try:
            store_util.barrier(store, 2, key_prefix, 1)
        except Exception as e:
            self.assertIsInstance(e, LookupError)
        store_util.barrier(store, 2, key_prefix, 1)

        store = MasterKVStore("dlrover/torch/test2")
        store.set_timeout(datetime.timedelta(seconds=1))
        key_prefix = "test"

        key = store_util._barrier_nonblocking(store, 2, key_prefix)
        try:
            store.get(key)
        except Exception as e:
            self.assertIsInstance(e, LookupError)
        key = store_util._barrier_nonblocking(store, 2, key_prefix)
        self.assertEqual("<val_ignored>", store.get(key))


class TrainingRdzvManagerFactoryTest(unittest.TestCase):
    def test_create_training_rdzv_manager_base(self):
        ctx = Context.singleton_instance()
        ctx.training_elastic_mode = "base"
        manager = create_training_rdzv_manager()
        self.assertIsInstance(manager, ElasticTrainingRendezvousManager)

    def test_create_training_rdzv_manager_ucp(self):
        ctx = Context.singleton_instance()
        ctx.training_elastic_mode = "ucp"
        manager = create_training_rdzv_manager()
        self.assertIsInstance(manager, UcpRdzvManager)

    def test_create_training_rdzv_manager_unknown(self):
        ctx = Context.singleton_instance()
        ctx.training_elastic_mode = "unknown"
        manager = create_training_rdzv_manager()
        self.assertIsInstance(manager, ElasticTrainingRendezvousManager)

    def test_ucp_rdzv_blocks_when_incomplete(self):
        manager = UcpRdzvManager()
        manager.set_rdzv_blocked(True)
        blocked, reason = manager.is_rdzv_blocked()
        self.assertTrue(blocked)
        self.assertTrue(reason)


class ElasticTrainingRendezvousManagerTest(unittest.TestCase):
    def test_rdzv_timeout(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager.update_rdzv_params(3, 3, 0.5, 1)
        rdzv_round = rdzv_manager.get_rdzv_round()
        self.assertEqual(rdzv_round, 0)
        self.assertEqual(rdzv_manager.rendezvous_events, {})
        rdzv_manager._alive_nodes = [0, 1, 2]
        rdzv_manager.join_rendezvous(0, 0, 8)
        rdzv_manager.join_rendezvous(1, 1, 8)
        time.sleep(1)
        round, _, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 0)
        self.assertDictEqual(world, {})
        self.assertEqual(len(rdzv_manager._waiting_nodes), 2)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 0)

        rdzv_manager = NetworkCheckRendezvousManager()
        rdzv_manager.update_rdzv_params(2, 2, 0.5, 1)
        rdzv_round = rdzv_manager.get_rdzv_round()
        self.assertEqual(rdzv_round, 0)
        self.assertEqual(rdzv_manager.rendezvous_events, {})
        rdzv_manager._alive_nodes = [0, 1]
        rdzv_manager.join_rendezvous(0, 0, 8)
        time.sleep(1)
        round, _, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 0)
        self.assertDictEqual(world, {})
        self.assertEqual(len(rdzv_manager._waiting_nodes), 1)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 0)

    def test_max_nodes(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager.update_rdzv_params(3, 3, 60, 1)
        rdzv_round = rdzv_manager.get_rdzv_round()
        self.assertEqual(rdzv_round, 0)
        self.assertEqual(rdzv_manager.rendezvous_events, {})
        rdzv_manager._alive_nodes = [0, 1, 2]

        # Register callback to verify _on_rdzv_completed is called
        callback_results = []
        rdzv_manager.add_rdzv_completed_callback(
            lambda rdzv_round, nodes: callback_results.append(
                (rdzv_round, nodes)
            )
        )

        rdzv_manager.join_rendezvous(0, 0, 8)
        rdzv_manager.join_rendezvous(1, 1, 8)
        round, _, world = rdzv_manager.get_comm_world(0)
        self.assertEqual(round, 0)
        self.assertDictEqual(world, {})
        self.assertEqual(len(rdzv_manager._waiting_nodes), 2)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 0)
        self.assertEqual(list(rdzv_manager.rendezvous_events.keys()), [0])
        # Callback should not be called before rdzv completes
        self.assertEqual(len(callback_results), 0)

        rdzv_manager.join_rendezvous(2, 2, 8)
        self.assertListEqual(list(rdzv_manager._node_rdzv_times), [0, 1, 2])
        for key in rdzv_manager._node_rdzv_times:
            self.assertLessEqual(rdzv_manager._node_rdzv_times[key], 0.05)
        round, _, world = rdzv_manager.get_comm_world(0)
        self.assertDictEqual(rdzv_manager._node_rdzv_times, {})
        self.assertEqual(round, 1)
        self.assertEqual(len(rdzv_manager._waiting_nodes), 0)
        self.assertEqual(len(rdzv_manager._rdzv_nodes), 3)
        self.assertListEqual(list(world.keys()), [0, 1, 2])
        self.assertEqual(list(rdzv_manager.rendezvous_events.keys()), [0])

        # Callback should be called exactly once with correct args
        self.assertEqual(len(callback_results), 1)
        self.assertEqual(callback_results[0][0], 0)  # finished_rdzv_round
        self.assertListEqual(callback_results[0][1], [0, 1, 2])  # node_ids

        # Subsequent get_comm_world calls should NOT trigger callback again
        round, _, world = rdzv_manager.get_comm_world(1)
        round, _, world = rdzv_manager.get_comm_world(2)
        self.assertEqual(len(callback_results), 1)

    def test_min_nodes(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager.update_rdzv_params(2, 3, 0.1, 1)

        # Register callback to verify _on_rdzv_completed
        callback_results = []
        rdzv_manager.add_rdzv_completed_callback(
            lambda rdzv_round, nodes: callback_results.append(
                (rdzv_round, nodes)
            )
        )

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

        # Callback called exactly once with correct round and node ids
        self.assertEqual(len(callback_results), 1)
        self.assertEqual(callback_results[0][0], 0)
        self.assertListEqual(callback_results[0][1], [0, 1])

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
        rdzv_manager._rdzv_params.max_nodes = 4
        rdzv_manager._waiting_nodes = {0: 0, 1: 1, 2: 2, 3: 3}
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [])

        rdzv_manager._rdzv_params.max_nodes = 5
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [4])

        rdzv_manager._rdzv_params.max_nodes = 3
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [])

        rdzv_manager._rdzv_params.max_nodes = 6
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [4, 5])

        rdzv_manager._rdzv_params.max_nodes = 4
        rdzv_manager._waiting_nodes = {}
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [0, 1, 2, 3])

        rdzv_manager._rdzv_params.max_nodes = 0
        self.assertEqual(rdzv_manager._get_lacking_ranks(), [])

    def test_multi_updating_waiting_nodes(self):
        rdzv_manager = ElasticTrainingRendezvousManager()

        join_num = 1000
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(join_num):
                executor.submit(
                    rdzv_manager.join_rendezvous,
                    i,
                    i,
                    8,
                )

        remove_num = 900
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(remove_num):
                node = Node("worker", i, name=f"worker-{i}", rank_index=i)
                executor.submit(
                    rdzv_manager.remove_alive_node,
                    node,
                )

        with ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(300):
                futures = [
                    executor.submit(
                        rdzv_manager.get_comm_world,
                        i,
                    )
                ]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        self.fail()

        time.sleep(5)
        self.assertEqual(
            len(rdzv_manager._waiting_nodes.keys()), join_num - remove_num
        )
        for i in rdzv_manager._waiting_nodes.keys():
            self.assertTrue(900 <= i <= 999)

    def test_on_rdzv_completed_callback_exception(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        results = []

        def bad_callback(rdzv_round, nodes):
            raise RuntimeError("callback error")

        def good_callback(rdzv_round, nodes):
            results.append((rdzv_round, nodes))

        rdzv_manager.add_rdzv_completed_callback(bad_callback)
        rdzv_manager.add_rdzv_completed_callback(good_callback)

        rdzv_manager._on_rdzv_completed(0, [0, 1, 2])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], (0, [0, 1, 2]))

    @patch("dlrover.python.master.elastic_training.rdzv_manager.job_ctx")
    def test_check_rdzv_completed_with_failed_node_before_relaunch(
        self, mock_job_ctx
    ):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager.update_rdzv_params(2, 2, 60, 1)

        # without node exit
        rdzv_manager._alive_nodes = [0, 1]
        rdzv_manager.join_rendezvous(0, 0, 8)
        rdzv_manager.join_rendezvous(1, 1, 8)
        self.assertTrue(rdzv_manager._check_rdzv_completed())

        # with node exit
        node = Node("worker", 1, name="worker-1", rank_index=1)
        node.reported_status = (NodeEventType.FAILED_EXITED, 1)
        mock_job_ctx.job_node.return_value = node
        self.assertFalse(rdzv_manager._check_rdzv_completed())


class NetworkCheckRendezvousManagerTest(unittest.TestCase):
    def test_network_check_rdzv(self):
        rdzv_manager = NetworkCheckRendezvousManager()
        rdzv_manager.update_rdzv_params(4, 4, 60, 1)
        rdzv_manager._alive_nodes = [0, 1, 2, 3]
        self.assertEqual(rdzv_manager.rendezvous_events, {})
        for i in range(4):
            round = rdzv_manager.join_rendezvous(i, i, 8)
        self.assertEqual(round, 0)
        self.assertEqual(list(rdzv_manager.rendezvous_events.keys()), [0])
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

    def test_process_error(self):
        rdzv_manager = ElasticTrainingRendezvousManager()
        rdzv_manager.update_rdzv_params(2, 2, 60, 1)
        rdzv_manager._alive_nodes = [0, 1]
        rdzv_manager.join_rendezvous(0, 0, 8)
        rdzv_manager.join_rendezvous(1, 1, 8)

        rdzv_manager.process_error(
            node_id=0,
            node_rank=0,
            err_type="timeout",
            err_message="Rendezvous timeout occurred",
            elapsed_time=30.5,
        )

        rdzv_manager._rdzv_round = 999
        rdzv_manager.process_error(
            node_id=1,
            node_rank=1,
            err_type="network_error",
            err_message="Network connection failed",
            elapsed_time=15.0,
        )


def _create_group_node_meta(node_id, node_rank, group, group_size, group_id):
    """Helper to create a NodeTopologyMeta with group info."""
    return NodeTopologyMeta(
        node_id=node_id,
        node_rank=node_rank,
        process_num=8,
        node_group=group,
        node_group_size=group_size,
        node_group_id=group_id,
    )


class GroupNodeNetworkCheckRendezvousManagerTest(unittest.TestCase):

    def _setup_manager(self, num_nodes, groups):
        """Setup a GroupNodeNetworkCheckRendezvousManager with group info.
        Args:
            num_nodes: total number of nodes.
            groups: dict mapping group_index -> list of node_ranks.
        """
        manager = GroupNodeNetworkCheckRendezvousManager()
        manager.update_rdzv_params(num_nodes, num_nodes, 60, 1)
        manager._alive_nodes = set(range(num_nodes))

        group_size = len(groups)
        for g_idx, ranks in groups.items():
            group_id = f"group-{g_idx}"
            for rank in ranks:
                meta = _create_group_node_meta(
                    rank, rank, g_idx, group_size, group_id
                )
                manager._waiting_nodes[rank] = meta

        # Simulate rdzv completion by directly populating _rdzv_nodes.
        from collections import OrderedDict

        manager._rdzv_nodes = OrderedDict()
        for rank in sorted(manager._waiting_nodes.keys()):
            manager._rdzv_nodes[rank] = manager._waiting_nodes[rank]
        manager._waiting_nodes = {}
        return manager

    def test_group_node_intra_pass_inter_pass(self):
        """2 groups, all checks pass: intra -> inter -> done."""
        # G0=[0,1,2,3], G1=[4,5,6,7]
        groups = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
        manager = self._setup_manager(8, groups)

        # Round 0: INTRA_INITIAL - adjacent pairing within groups.
        node_groups = manager._group_nodes(0)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTRA_INITIAL)
        # G0: {0,1}, {2,3}, G1: {4,5}, {6,7}
        self.assertEqual(len(node_groups), 4)
        self.assertListEqual(sorted(node_groups[0].keys()), [0, 1])
        self.assertListEqual(sorted(node_groups[1].keys()), [2, 3])
        self.assertListEqual(sorted(node_groups[2].keys()), [4, 5])
        self.assertListEqual(sorted(node_groups[3].keys()), [6, 7])

        # All nodes pass intra check.
        for i in range(8):
            manager._node_status[i] = True
            manager._node_times[i] = 1.0

        # Round 1: INTER_INITIAL - same-position cross-group pairing.
        node_groups = manager._group_nodes(1)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTER_INITIAL)
        # {0,4}, {1,5}, {2,6}, {3,7}
        self.assertEqual(len(node_groups), 4)
        self.assertListEqual(sorted(node_groups[0].keys()), [0, 4])
        self.assertListEqual(sorted(node_groups[1].keys()), [1, 5])
        self.assertListEqual(sorted(node_groups[2].keys()), [2, 6])
        self.assertListEqual(sorted(node_groups[3].keys()), [3, 7])

        # All nodes pass inter check -> check_fault_node returns no faults.
        manager._reported_nodes = set(range(8))
        for i in range(8):
            manager._node_status[i] = True
        nodes, reason = manager.check_fault_node()
        self.assertListEqual(nodes, [])
        # Round should jump to end of cycle.
        self.assertEqual(manager._rdzv_round % manager._check_round, 0)

    def test_group_node_intra_fail_diagnostic(self):
        """2 groups, intra check fails -> diagnostic identifies fault."""
        groups = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
        manager = self._setup_manager(8, groups)

        # Round 0: INTRA_INITIAL.
        manager._group_nodes(0)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTRA_INITIAL)

        # Node 1 fails in intra check.
        for i in range(8):
            manager._node_status[i] = True
            manager._node_times[i] = 1.0
        manager._node_status[0] = False
        manager._node_status[1] = False
        manager._node_times[1] = 5.0  # Slower (likely faulty).

        # Round 1: should be INTRA_DIAGNOSTIC.
        node_groups = manager._group_nodes(1)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTRA_DIAGNOSTIC)
        # Cross pairing within groups: sorted by time.
        # G0 sorted by time: [0(1.0), 2(1.0), 3(1.0), 1(5.0)]
        # Pairs: {0,1}, {2,3} (fastest with slowest).
        for group in node_groups:
            self.assertLessEqual(len(group), 2)

        # Simulate diagnostic: node 1 fails again.
        manager._reported_nodes = set(range(8))
        for i in range(8):
            manager._node_status[i] = True
        manager._node_status[1] = False
        nodes, reason = manager.check_fault_node()
        self.assertIn(1, nodes)
        self.assertEqual(reason, NetworkFailureReason.NODE_FAILURE)

    def test_group_node_inter_fail_diagnostic(self):
        """2 groups, intra passes but inter fails -> inter diagnostic."""
        groups = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
        manager = self._setup_manager(8, groups)

        # Round 0: INTRA_INITIAL - all pass.
        manager._group_nodes(0)
        for i in range(8):
            manager._node_status[i] = True
            manager._node_times[i] = 1.0

        # check_fault_node should return NEXT_PHASE.
        manager._reported_nodes = set(range(8))
        nodes, reason = manager.check_fault_node()
        self.assertListEqual(nodes, [])
        self.assertEqual(reason, NetworkFailureReason.NEXT_PHASE)

        # Round 1: INTER_INITIAL - node 4 fails.
        manager._fault_nodes.clear()
        node_groups = manager._group_nodes(1)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTER_INITIAL)

        manager._reported_nodes = set(range(8))
        for i in range(8):
            manager._node_status[i] = True
            manager._node_times[i] = 1.0
        manager._node_status[0] = False
        manager._node_status[4] = False
        manager._node_times[4] = 5.0

        # Round 2: INTER_DIAGNOSTIC.
        node_groups = manager._group_nodes(2)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTER_DIAGNOSTIC)
        # Shifted pairing: G0 shift 0, G1 shift 1.
        # The pairing should be different from INTER_INITIAL.
        self.assertTrue(len(node_groups) > 0)
        # Each group should contain nodes from different groups.
        for group in node_groups:
            group_indices = set()
            for rank in group:
                group_indices.add(manager._rdzv_nodes[rank].node_group)
            self.assertTrue(len(group_indices) > 1)

    def test_group_node_4_groups(self):
        """4 groups with 2 nodes each."""
        # G0=[0,1], G1=[2,3], G2=[4,5], G3=[6,7]
        groups = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
        manager = self._setup_manager(8, groups)

        # Round 0: INTRA_INITIAL.
        node_groups = manager._group_nodes(0)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTRA_INITIAL)
        # Each group has 1 pair: {0,1}, {2,3}, {4,5}, {6,7}
        self.assertEqual(len(node_groups), 4)
        self.assertListEqual(sorted(node_groups[0].keys()), [0, 1])
        self.assertListEqual(sorted(node_groups[1].keys()), [2, 3])

        # All pass intra.
        for i in range(8):
            manager._node_status[i] = True
            manager._node_times[i] = 1.0

        # Round 1: INTER_INITIAL.
        node_groups = manager._group_nodes(1)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTER_INITIAL)
        # Position 0: {0,2,4,6}, Position 1: {1,3,5,7}
        self.assertEqual(len(node_groups), 2)
        self.assertListEqual(sorted(node_groups[0].keys()), [0, 2, 4, 6])
        self.assertListEqual(sorted(node_groups[1].keys()), [1, 3, 5, 7])

        # All pass inter -> done.
        manager._reported_nodes = set(range(8))
        for i in range(8):
            manager._node_status[i] = True
        nodes, reason = manager.check_fault_node()
        self.assertListEqual(nodes, [])

    def test_group_node_4_groups_inter_diagnostic(self):
        """4 groups with 2 nodes each, inter diagnostic phase."""
        groups = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}
        manager = self._setup_manager(8, groups)

        # All pass intra.
        for i in range(8):
            manager._node_status[i] = True
            manager._node_times[i] = 1.0

        # Round 1: INTER_INITIAL.
        manager._group_nodes(1)

        # Some inter failures.
        for i in range(8):
            manager._node_status[i] = True
            manager._node_times[i] = 1.0
        manager._node_status[2] = False
        manager._node_times[2] = 5.0

        # Round 2: INTER_DIAGNOSTIC - shifted pairing.
        node_groups = manager._group_nodes(2)
        self.assertEqual(manager._current_phase, GroupNodeCheckPhase.INTER_DIAGNOSTIC)
        self.assertTrue(len(node_groups) > 0)
        # Each group should have nodes from multiple groups.
        for group in node_groups:
            group_indices = set()
            for rank in group:
                group_indices.add(manager._rdzv_nodes[rank].node_group)
            self.assertTrue(len(group_indices) > 1)

    def test_group_node_fallback_without_group_info(self):
        """Without group info, should fall back to base behavior."""
        manager = GroupNodeNetworkCheckRendezvousManager()
        manager.update_rdzv_params(4, 4, 60, 1)
        manager._alive_nodes = set(range(4))
        for i in range(4):
            # No group info (default node_group=-1).
            meta = NodeTopologyMeta(
                node_id=i, node_rank=i, process_num=8
            )
            manager._waiting_nodes[i] = meta

        from collections import OrderedDict

        manager._rdzv_nodes = OrderedDict()
        for rank in sorted(manager._waiting_nodes.keys()):
            manager._rdzv_nodes[rank] = manager._waiting_nodes[rank]
        manager._waiting_nodes = {}

        # Should fall back to base class grouping.
        node_groups = manager._group_nodes(0)
        self.assertEqual(len(node_groups), 2)
        self.assertListEqual(sorted(node_groups[0].keys()), [0, 1])
        self.assertListEqual(sorted(node_groups[1].keys()), [2, 3])
