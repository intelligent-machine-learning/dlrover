# Copyright 2026 The DLRover Authors. All rights reserved.
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
import importlib
import json
import os
import sys
import time
import unittest
from datetime import datetime
from typing import List
from unittest import mock
from unittest.mock import MagicMock, patch

from dlrover.python.common import comm
from dlrover.python.common.comm import (
    BaseRequest,
    BaseResponse,
    DiagnosisAction,
    HeartbeatResponse,
)
from dlrover.python.common.constants import (
    CommunicationType,
    NodeEnv,
    NodeEventType,
    NodeType,
    RendezvousName,
    TrainingExceptionLevel,
)
from dlrover.python.common.event.context import JobEventContext
from dlrover.python.common.event.train_event import (
    TrainEventName,
    TrainEventState,
)
from dlrover.python.common.global_context import Context
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
    JobAbortionAction,
)
from dlrover.python.diagnosis.datacollector.atorch_event_collector import (
    AtorchEventCollector,
)
from dlrover.python.elastic_agent.master_client import (
    RayMasterClient,
    build_master_client,
)
from dlrover.python.tests.test_utils import start_local_master
from dlrover.python.training_event.event import EventTargetName, EventTypeName

_event_context = JobEventContext.singleton_instance()


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

    def test_kv_store(self):
        self._master_client.kv_store_set("alpha", b"0")
        self.assertEqual(self._master_client.kv_store_get("alpha"), b"0")
        self._master_client.kv_store_add("beta", 1)
        self.assertEqual(self._master_client.kv_store_get("beta"), 1)
        self._master_client.kv_store_add("beta", 10)
        self.assertEqual(self._master_client.kv_store_get("beta"), 11)

        self.assertDictEqual(
            self._master_client.kv_store_multi_get(["alpha", "beta"]),
            {"alpha": b"0", "beta": 11},
        )
        self.assertDictEqual(
            self._master_client.kv_store_multi_get(["omega"]), {}
        )
        self._master_client.kv_store_multi_set(
            ["alpha", "beta", "gamma"], [b"0", b"100", b"200"]
        )
        self.assertDictEqual(
            self._master_client.kv_store_multi_get(["alpha", "beta", "gamma"]),
            {"alpha": b"0", "beta": b"100", "gamma": b"200"},
        )

    def test_num_nodes_waiting(self):
        rdzv_name = RendezvousName.TRAINING
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

    def test_event_log(self):
        _event_context.train_steps.clear_step_events()
        _event_context.ckpt_steps.clear_step_events()

        collector = AtorchEventCollector(
            filepath="dlrover/python/tests/data/training_events",
            local_world_size=1,
            retry_timeout=1,
        )
        collector._client = self._master_client
        collector.reset_first_step()
        collector.start_collectors()
        time.sleep(1)
        collector.stop_collectors()

        self.assertEqual(_event_context.train_steps.size(), 2)
        first_step = _event_context.train_steps.get_first_step_event()
        last_step = _event_context.train_steps.get_last_step_event()
        self.assertEqual(first_step.step, 177020)
        self.assertEqual(last_step.step, 177021)

        self.assertEqual(_event_context.ckpt_steps.size(), 2)
        first_step = _event_context.ckpt_steps.get_first_step_event()
        last_step = _event_context.ckpt_steps.get_last_step_event()
        self.assertEqual(first_step.step, 177215)
        self.assertEqual(last_step.step, 177413)

    def test_report_atorch_events(self):
        _event_context.train_steps.clear_step_events()
        _event_context.ckpt_steps.clear_step_events()

        now = int(datetime.now().timestamp())
        self._master_client.report_atorch_event(
            event_ts=now,
            event_target=EventTargetName.TRAINER,
            event_name=TrainEventName.TRAIN_EVT_STEP,
            event_type=EventTypeName.BEGIN,
            event_step=1,
        )
        last_step = _event_context.train_steps.get_last_step_event()
        self.assertEqual(last_step.step, 1)
        self.assertEqual(
            last_step.event_state, TrainEventState.TRAIN_EVT_BEGIN
        )
        self.assertEqual(last_step.begin_timestamp, now)

        self._master_client.report_atorch_event(
            event_ts=now + 1,
            event_target=EventTargetName.TRAINER,
            event_name=TrainEventName.TRAIN_EVT_STEP,
            event_type=EventTypeName.END,
            event_step=1,
        )
        last_step = _event_context.train_steps.get_last_step_event()
        self.assertEqual(last_step.step, 1)
        self.assertEqual(
            last_step.event_state, TrainEventState.TRAIN_EVT_BEGIN
        )
        self.assertEqual(last_step.begin_timestamp, now)

        self._master_client.report_atorch_event(
            event_ts=now + 1,
            event_target=EventTargetName.TRAINER,
            event_name=TrainEventName.TRAIN_EVT_STEP,
            event_type=EventTypeName.END,
            event_step=2,
        )
        last_step = _event_context.train_steps.get_last_step_event()
        self.assertEqual(last_step.step, 2)
        self.assertEqual(last_step.event_state, TrainEventState.TRAIN_EVT_END)
        self.assertEqual(last_step.begin_timestamp, now)
        self.assertEqual(last_step.end_timestamp, now + 1)
        self.assertEqual(last_step.step_time, 1)

        self._master_client.report_atorch_event(
            event_ts=now + 5,
            event_target=EventTargetName.TRAINER,
            event_name=TrainEventName.TRAIN_EVT_STEP,
            event_type=EventTypeName.BEGIN,
            event_step=2,
        )
        self.assertEqual(_event_context.train_steps.size(), 2)

        now = int(datetime.now().timestamp())
        self._master_client.report_atorch_event(
            event_ts=now + 100,
            event_target=EventTargetName.TRAINER,
            event_name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            event_type=EventTypeName.BEGIN,
            event_step=100,
        )
        self.assertEqual(_event_context.ckpt_steps.size(), 1)
        last_step = _event_context.ckpt_steps.get_last_step_event()
        self.assertEqual(last_step.step, 100)
        self.assertEqual(
            last_step.event_state, TrainEventState.TRAIN_EVT_BEGIN
        )
        self.assertEqual(last_step.begin_timestamp, now + 100)
        self.assertEqual(last_step.end_timestamp, 0)

        self._master_client.report_atorch_event(
            event_ts=now + 150,
            event_target=EventTargetName.TRAINER,
            event_name=TrainEventName.TRAIN_EVT_FLASH_CKPT,
            event_type=EventTypeName.END,
            event_step=100,
        )
        self.assertEqual(_event_context.ckpt_steps.size(), 1)
        last_step = _event_context.ckpt_steps.get_last_step_event()
        self.assertEqual(last_step.step, 100)
        self.assertEqual(last_step.event_state, TrainEventState.TRAIN_EVT_END)
        self.assertEqual(last_step.begin_timestamp, now + 100)
        self.assertEqual(last_step.end_timestamp, now + 150)
        self.assertEqual(last_step.step_time, 50)

    @patch.dict("sys.modules", {"dlrover.proto": None})
    def test_pb_not_installed(self):
        module = "dlrover.python.elastic_agent.master_client"
        if module in sys.modules:
            del sys.modules[module]

        with self.assertRaises(ImportError):
            importlib.import_module(
                "dlrover.python.elastic_agent.master_client.GrpcMasterClient"
            )

    def test_report_action(self):
        self._master_client.report_action(JobAbortionAction(reason="test"))


class MasterClientBuildTest(unittest.TestCase):
    def test_build_failure(self):
        master_client = build_master_client("test", 1)
        self.assertIsNone(master_client)


class MasterHttpClientTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ[NodeEnv.DLROVER_MASTER_SERVICE_TYPE] = (
            CommunicationType.COMM_SERVICE_HTTP
        )
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
        rdzv_name = RendezvousName.TRAINING
        num = self._master_client.num_nodes_waiting(rdzv_name)
        self.assertEqual(num, 0)

        # report request
        res = self._master_client.ready_for_ps_relaunch()
        self.assertTrue(res.success)


class MasterRayClientTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ[NodeEnv.DLROVER_MASTER_ADDR] = "test_id"
        os.environ[NodeEnv.DLROVER_MASTER_SERVICE_TYPE] = (
            CommunicationType.COMM_SERVICE_RAY
        )
        context = Context.singleton_instance()
        context.master_service_type = "ray"
        self._master_client = build_master_client(None, 3)

    def tearDown(self) -> None:
        os.environ.clear()
        context = Context.singleton_instance()
        context.master_service_type = "grpc"

    @patch.dict("sys.modules", {"ray": None})
    def test_ray_not_installed(self):
        module = "dlrover.python.elastic_agent.master_client"
        if module in sys.modules:
            del sys.modules[module]

        with self.assertRaises(ImportError):
            importlib.import_module(
                "dlrover.python.elastic_agent.master_client.RayMasterClient"
            )

    @patch("ray.get")
    def test_ray_client(self, mock_get):
        self.assertIsNotNone(self._master_client)
        self.assertTrue(isinstance(self._master_client, RayMasterClient))

        self.assertIsNone(self._master_client._master_actor_handle)
        self._master_client._master_actor_handle = "test"
        self.assertEqual(
            self._master_client._get_master_actor_handle(), "test"
        )
        self.assertFalse(self._master_client.get_elastic_run_config())

        req = BaseRequest(node_id=0, node_type="test")
        mock_get.return_value = True
        self._master_client._master_actor_handle = MagicMock()
        self._master_client._master_actor_handle.agent_report = MagicMock()
        self._master_client._master_actor_handle.agent_report.remote = (
            MagicMock()
        )
        self.assertTrue(self._master_client._report(req))

        rep = BaseResponse()
        mock_get.return_value = rep
        self._master_client._master_actor_handle.agent_report = MagicMock()
        self._master_client._master_actor_handle.agent_get.remote = MagicMock()
        self.assertFalse(self._master_client._get(req))
