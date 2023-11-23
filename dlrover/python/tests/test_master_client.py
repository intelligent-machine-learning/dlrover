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

import json
import time
import unittest

from dlrover.python.common import grpc
from dlrover.python.common.constants import (
    NodeStatus,
    NodeType,
    TrainingMsgLevel,
)
from dlrover.python.elastic_agent.master_client import build_master_client
from dlrover.python.tests.test_utils import start_local_master


class MasterClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        self._master_client = build_master_client(addr, 0.5)

    def tearDown(self):
        self._master.stop()

    def test_open_channel(self):
        self.assertEqual(self._master_client._timeout, 0.5)
        self._master_client.close_channel()
        self._master_client.open_channel()

    def test_report_used_resource(self):
        gpu_stats: list[grpc.GPUStats] = [
            grpc.GPUStats(
                index=0,
                total_memory_mb=24000,
                used_memory_mb=4000,
                gpu_utilization=55.5,
            )
        ]
        result = self._master_client.report_used_resource(1024, 10, gpu_stats)
        self.assertTrue(result.success)

    def test_report_failures(self):
        res = self._master_client.report_failures(
            "test", 0, TrainingMsgLevel.WARNING
        )
        self.assertIsNone(res)

    def test_ready_for_ps_relaunch(self):
        res = self._master_client.ready_for_ps_relaunch()
        self.assertTrue(res.success)

    def test_get_shard_checkpoint(self):
        ds_name = "test"
        self._master_client.report_dataset_shard_params(
            batch_size=64,
            num_epochs=1,
            dataset_size=10000,
            shuffle=False,
            num_minibatches_per_shard=10,
            dataset_name=ds_name,
            task_type=0,
            storage_type="",
        )
        _, task = self._master_client.get_task(ds_name)
        self._master_client.report_task_result(ds_name, task.task_id, "")
        _, task = self._master_client.get_task(ds_name)
        checkpoint = self._master_client.get_shard_checkpoint(ds_name)
        checkpoint = json.loads(checkpoint)
        self.assertListEqual(checkpoint["doing"][0], [640, 1280])
        self.assertEqual(len(checkpoint["todo"]), 14)
        checkpoint["doing"][0] = [1280, 1600]
        checkpoint = json.dumps(checkpoint)
        self._master_client.report_shard_checkpoint(checkpoint)
        dataset = self._master.task_manager.get_dataset(ds_name)
        task = dataset.todo[0]
        self.assertEqual(task.shard.start, 1280)
        self.assertEqual(task.shard.end, 1600)

    def test_get_cluster_version(self):
        self._master_client.update_cluster_version(
            "LOCAL",
            1,
            NodeType.WORKER,
            0,
        )
        version = self._master_client.get_cluster_version(
            "LOCAL", NodeType.WORKER, 0
        )
        self.assertEqual(version, 0)

    def test_report(self):
        res = self._master_client.update_node_addr(
            NodeType.PS, 0, "127.0.0.1:1234"
        )
        self.assertTrue(res.success, True)

        res = self._master_client.update_node_event(NodeType.PS, 0, "ADDED")
        self.assertTrue(res.success, True)

        ts = int(time.time())
        self._master_client.report_global_step(100, ts)

        model_info = grpc.ModelInfo()
        self._master_client.report_model_info(model_info)

        success = self._master_client.join_sync("test-sync")
        self.assertFalse(success)

        success = self._master_client.sync_finished("test-sync")
        self.assertFalse(success)

        success = self._master_client.barrier("test-barrier", True)
        self.assertFalse(success)

        self._master_client.report_network_status(0, NodeStatus.SUCCEEDED, 10)

    def test_get(self):
        nodes, failure = self._master_client.query_ps_nodes()
        self.assertEqual(len(nodes), 0)
        self.assertFalse(failure)

        status = self._master_client.query_training_status()
        self.assertEqual(status, 3)

        nodes = self._master_client.get_running_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].type, NodeType.WORKER)

        nodes = self._master_client.check_fault_node()
        self.assertListEqual(nodes, [])

        round = self._master_client.join_rendezvous(0, 8, "elastic-training")
        self.assertEqual(round, 0)

        config = self._master_client.get_paral_config()
        if config:
            self.assertIsInstance(config, grpc.ParallelConfig)

    def test_num_nodes_waiting(self):
        rdzv_name = object()
        num = self._master_client.num_nodes_waiting(rdzv_name)
        self.assertEqual(num, 0)
