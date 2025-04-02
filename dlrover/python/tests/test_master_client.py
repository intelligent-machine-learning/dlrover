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
import os
import time
import unittest
from typing import List
from unittest import mock

from dlrover.python.common import comm
from dlrover.python.common.comm import DiagnosisAction, HeartbeatResponse
from dlrover.python.common.constants import (
    CommunicationType,
    NodeEnv,
    NodeEventType,
    NodeType,
    RendezvousName,
    TrainingExceptionLevel,
)
from dlrover.python.common.global_context import Context
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
)
from dlrover.python.elastic_agent.master_client import build_master_client
from dlrover.python.tests.test_utils import start_local_master


class MasterClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        self._master_client = build_master_client(addr, 1)

    def tearDown(self):
        self._master.stop()

    def test_open_channel(self):
        self.assertEqual(self._master_client._timeout, 1)
        self.assertEqual(self._master_client._timeout, 1)
        self._master_client._close_grpc_channel()
        self._master_client._open_grpc_channel()

    def test_report_used_resource(self):
        gpu_stats: List[comm.GPUStats] = [
            comm.GPUStats(
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
            "test", 0, TrainingExceptionLevel.WARNING
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

        res = self._master_client.report_node_event(
            NodeEventType.ADDED, "ADDED"
        )
        self.assertTrue(res.success, True)

        res = self._master_client.report_node_event(
            NodeEventType.SUCCEEDED_EXITED, "SUCCEEDED_EXITED"
        )
        self.assertTrue(res.success, True)

        ts = int(time.time())
        self._master_client.report_global_step(100, ts)

        model_info = comm.ModelInfo()
        self._master_client.report_model_info(model_info)

        success = self._master_client.join_sync("test-sync")
        self.assertFalse(success)

        success = self._master_client.sync_finished("test-sync")
        self.assertFalse(success)

        success = self._master_client.barrier("test-barrier", True)
        self.assertFalse(success)

        self._master_client.report_network_check_status(
            0, NodeEventType.NODE_CHECK_SUCCEEDED, 10
        )

        success = self._master_client.sync_checkpoint(100)
        self.assertFalse(success)

    def test_get(self):
        nodes, failure = self._master_client.query_ps_nodes()
        self.assertEqual(len(nodes), 0)
        self.assertFalse(failure)

        status = self._master_client.query_training_status()
        self.assertEqual(status, 3)

        nodes = self._master_client.get_running_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0].type, NodeType.WORKER)

        nodes, _ = self._master_client.check_fault_node(timeout=1)
        self.assertListEqual(nodes, [])

        round = self._master_client.join_rendezvous(0, 8, "elastic-training")
        self.assertEqual(round, 0)

        config = self._master_client.get_paral_config()
        if config:
            self.assertIsInstance(config, comm.ParallelConfig)

    def test_num_nodes_waiting(self):
        rdzv_name = RendezvousName.ELASTIC_TRAINING
        num = self._master_client.num_nodes_waiting(rdzv_name)
        self.assertEqual(num, 0)

    def test_report_heartbeat(self):
        now = time.time()
        self._master_client._get = mock.MagicMock(side_effect=[None])
        action = self._master_client.report_heart_beat(now)
        self.assertTrue(isinstance(action, NoAction))

        event_action = EventAction(
            "info", "job", "test", "test123", {"k1": "v1"}
        )
        action_dto = DiagnosisAction(
            action_cls=event_action.__class__.__name__,
            action_content=event_action.to_json(),
        )
        response_dto = HeartbeatResponse(action_dto)
        self._master_client._get = mock.MagicMock(return_value=response_dto)
        action = self._master_client.report_heart_beat(now)
        self.assertTrue(isinstance(action, EventAction))


class MasterClientBuildTest(unittest.TestCase):
    def test_build_failure(self):
        master_client = build_master_client("test", 1)
        self.assertIsNone(master_client)


class MasterHttpClientTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ[
            NodeEnv.DLROVER_MASTER_SERVICE_TYPE
        ] = CommunicationType.COMM_SERVICE_HTTP
        context = Context.singleton_instance()
        context.master_service_type = "http"
        self._master, addr = start_local_master()
        self._master_client = build_master_client(addr, 3)

    def tearDown(self):
        self._master.stop()
        context = Context.singleton_instance()
        context.master_service_type = "grpc"
        os.environ.clear()

    def test_http_client(self):
        # get request
        rdzv_name = RendezvousName.ELASTIC_TRAINING
        num = self._master_client.num_nodes_waiting(rdzv_name)
        self.assertEqual(num, 0)

        # report request
        res = self._master_client.ready_for_ps_relaunch()
        self.assertTrue(res.success)
