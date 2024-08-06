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

import time
import unittest
from unittest import mock

import ray

from dlrover.proto import elastic_training_pb2
from dlrover.python.common import grpc
from dlrover.python.common.constants import (
    NodeStatus,
    NodeType,
    PSClusterVersionType,
    RendezvousName,
)
from dlrover.python.common.grpc import GPUStats
from dlrover.python.master.diagnosis.diagnosis import DiagnosisManager
from dlrover.python.master.elastic_training.elastic_ps import ElasticPsService
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
)
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.dist_job_manager import create_job_manager
from dlrover.python.master.servicer import MasterServicer
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.tests.test_utils import (
    MockK8sPSJobArgs,
    MockRayJobArgs,
    mock_k8s_client,
)
from dlrover.python.util.queue.queue import RayEventQueue

ray_event_queue = RayEventQueue.singleton_instance()


class MasterServicerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        params = MockK8sPSJobArgs()
        params.initilize()
        worker_resource = params.node_args[NodeType.WORKER].group_resource
        worker_resource.node_resource.gpu_num = 1
        worker_resource.node_resource.gpu_type = "a100"
        speed_monitor = SpeedMonitor()
        self.task_manager = TaskManager(False, speed_monitor)
        self.job_manager = create_job_manager(params, speed_monitor)
        self.job_manager._init_nodes()
        self.job_manager._init_job_auto_scaler()
        for node in self.job_manager._job_nodes[NodeType.WORKER].values():
            node.status = NodeStatus.RUNNING
        self.job_metric_collector = JobMetricCollector(
            "1", "default", "local", "dlrover"
        )
        self.elastic_ps_service = ElasticPsService()
        training_manager = ElasticTrainingRendezvousManager()
        rdzv_managers = {
            RendezvousName.ELASTIC_TRAINING: training_manager,
            RendezvousName.NETWORK_CHECK: NetworkCheckRendezvousManager(),
        }
        sync_service = SyncService(self.job_manager)
        self.servicer = MasterServicer(
            task_manager=self.task_manager,
            job_manager=self.job_manager,
            speed_monitor=speed_monitor,
            rdzv_managers=rdzv_managers,
            diagnosis_manager=DiagnosisManager(),
            job_metric_collector=self.job_metric_collector,
            elastic_ps_service=self.elastic_ps_service,
            sync_service=sync_service,
        )

    def test_query_running_nodes(self):
        request = elastic_training_pb2.Message()
        message = grpc.RunningNodesRequest()
        request.data = message.serialize()
        res = self.servicer.get(request, None)
        ret: grpc.RunningNodes = grpc.deserialize_message(res.data)
        self.assertEqual(len(ret.nodes), 3)

    def test_dataset_service(self):
        request = grpc.DatasetShardParams()
        request.batch_size = 10
        request.num_epochs = 1
        request.dataset_size = 1000
        request.shuffle = False
        request.num_minibatches_per_shard = 10
        request.dataset_name = "test"
        request.task_type = elastic_training_pb2.TRAINING
        request.storage_type = "text"
        self.servicer._collect_dataset_shard_params(request)

        collector = self.job_metric_collector._stats_reporter
        self.assertEqual(collector._dataset_metric.get_size(), 1000)

        request = grpc.TaskRequest("test")
        task: grpc.Task = self.servicer._get_task(NodeType.WORKER, 0, request)
        self.assertEqual(task.task_id, 0)
        self.assertEqual(task.shard.start, 0)
        self.assertEqual(task.shard.end, 100)

        request = grpc.TaskResult(dataset_name="test", task_id=0)
        request.task_id = 0
        request.dataset_name = "test"
        self.servicer._report_task_result(request)
        self.assertEqual(len(self.task_manager._datasets["test"].doing), 0)

        request = grpc.ShardCheckpointRequest("test")
        request.dataset_name = "test"

        checkpoint = self.servicer._get_shard_checkpoint(request)
        self.assertLessEqual(10, len(checkpoint.content))
        self.servicer._restore_shard_checkpoint(checkpoint)

    def test_metric_service(self):
        self.job_manager._init_nodes()
        self.job_manager._init_job_auto_scaler()
        request = grpc.ResourceStats(gpu_stats=[])
        request.memory = 4096
        request.cpu = 2
        gpu_stats: list[GPUStats] = [
            GPUStats(
                index=0,
                total_memory_mb=24000,
                used_memory_mb=4000,
                gpu_utilization=55.5,
            )
        ]
        for gpu in gpu_stats:
            gpu_stats_message = grpc.GPUStats()
            gpu_stats_message.index = gpu.index
            gpu_stats_message.total_memory_mb = gpu.total_memory_mb
            gpu_stats_message.used_memory_mb = gpu.used_memory_mb
            gpu_stats_message.gpu_utilization = gpu.gpu_utilization
            request.gpu_stats.append(gpu_stats_message)
        self.servicer._update_node_resource_usage(NodeType.WORKER, 0, request)
        self.servicer._update_node_resource_usage(NodeType.PS, 0, request)

        request = grpc.ModelInfo(
            tensor_stats=grpc.TensorStats(), op_stats=grpc.OpStats()
        )
        request.tensor_stats.variable_count = 100
        request.tensor_stats.total_variable_size = 10000

        request.op_stats.op_count = 100
        request.op_stats.flops = 10000
        self.servicer._collect_model_info(request)
        reporter = self.job_metric_collector._stats_reporter
        reporter._runtime_stats = []
        self.assertEqual(reporter._model_info.op_stats.flops, 10000)

        worker0 = self.job_manager._job_nodes[NodeType.WORKER][0]
        worker0.status = NodeStatus.RUNNING
        ps0 = self.job_manager._job_nodes[NodeType.PS][0]
        ps0.status = NodeStatus.RUNNING
        request = grpc.GlobalStep()
        self.task_manager._speed_monitor.add_running_worker(NodeType.WORKER, 0)
        self.task_manager._speed_monitor.set_target_worker_num(1)
        ts = int(time.time())
        request.timestamp = ts
        request.step = 100
        self.servicer._collect_global_step(request)
        request.timestamp = ts + 10
        request.step = 1100
        self.servicer._collect_global_step(request)
        self.job_metric_collector._report_runtime_stats()
        self.assertEqual(len(reporter._runtime_stats), 2)
        self.assertEqual(reporter._runtime_stats[0].global_step, 1100)
        self.assertEqual(len(reporter._runtime_stats[0].running_nodes), 2)

        request.timestamp = ts + 20
        request.step = 2100
        self.servicer._collect_global_step(request)

        request.timestamp = ts + 30
        request.step = 3100
        self.servicer._collect_global_step(request)

        request.timestamp = ts + 40
        request.step = 4100
        self.servicer._collect_global_step(request)

        request.timestamp = ts + 50
        request.step = 5100
        self.servicer._collect_global_step(request)
        self.assertTrue(self.servicer._start_autoscale)

    def test_query_ps_nodes(self):
        self.job_manager._init_nodes()
        for node in self.job_manager._job_nodes[NodeType.PS].values():
            node.status = NodeStatus.RUNNING
        res = self.servicer._query_ps_nodes()
        self.assertEqual(len(res.nodes), 3)
        self.assertEqual(
            res.nodes[0].addr, "test-edljob-ps-0.default.svc:2222"
        )

    def test_get(self):
        request = elastic_training_pb2.Message()
        request.data = b""
        response = self.servicer.get(request, None)
        self.assertEqual(response.data, b"")

    def test_get_cluster_version(self):
        message = grpc.ClusterVersionRequest(NodeType.WORKER, 0, "local")
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        response = self.servicer.get(request, None)
        res_msg = grpc.deserialize_message(response.data)
        self.assertEqual(res_msg.version, 0)

        message = grpc.ClusterVersionRequest(NodeType.PS, 0, "local")
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        response = self.servicer.get(request, None)
        res_msg = grpc.deserialize_message(response.data)
        self.assertEqual(res_msg.version, 0)

    def test_get_training_status(self):
        message = grpc.TrainingStatusRequest()
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        response = self.servicer.get(request, None)
        res_msg: grpc.TrainingStatus = grpc.deserialize_message(response.data)
        self.assertEqual(res_msg.status, 3)

    def test_num_nodes_waiting(self):
        message = grpc.WaitingNodeNumRequest(
            0, 8, RendezvousName.ELASTIC_TRAINING
        )
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        self.servicer._rdzv_managers[
            RendezvousName.ELASTIC_TRAINING
        ]._waiting_nodes = {0: 8}
        response = self.servicer.get(request, None)
        res_msg: grpc.RendezvousState = grpc.deserialize_message(response.data)
        self.assertEqual(res_msg.waiting_num, 1)

    def test_report(self):
        request = elastic_training_pb2.Message()
        request.data = b""
        response = self.servicer.report(request, None)
        self.assertFalse(response.success)

    def test_report_task_result(self):
        request = elastic_training_pb2.Message()
        message = grpc.TaskResult("test", 0, "error")
        dataset_params = grpc.DatasetShardParams(
            batch_size=64,
            num_epochs=1,
            dataset_size=10000,
            num_minibatches_per_shard=10,
            dataset_name="test",
            task_type=elastic_training_pb2.PREDICTION,
        )
        self.servicer._collect_dataset_shard_params(dataset_params)
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertFalse(response.success, False)

        message = grpc.TaskResult("test", 0, "")
        request.data = message.serialize()
        self.servicer._start_autoscale = False
        self.servicer._speed_monitor.completed_global_step == 0
        self.servicer._start_training_time = time.time() - 3600
        response = self.servicer.report(request, None)
        self.assertTrue(self.servicer._start_autoscale)
        self.assertTrue(response.success)

    def test_update_cluster_version(self):
        request = elastic_training_pb2.Message()
        message = grpc.ClusterVersion(
            NodeType.WORKER, 0, PSClusterVersionType.LOCAL, 1
        )
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)
        self.assertEqual(
            self.servicer._elastic_ps_service._worker_local_version[0], 1
        )

        message = grpc.ClusterVersion(
            NodeType.WORKER, 0, PSClusterVersionType.RESTORED, 1
        )
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)
        self.assertEqual(
            self.servicer._elastic_ps_service._worker_restored_version[0], 1
        )

        message = grpc.ClusterVersion(
            NodeType.PS, 0, PSClusterVersionType.GLOBAL, 1
        )
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)
        self.assertEqual(self.servicer._elastic_ps_service._global_version, 1)

    def test_sync(self):
        request = elastic_training_pb2.Message()
        message = grpc.SyncJoin("test")
        request.data = message.serialize()
        request.node_type = NodeType.WORKER
        request.node_id = 0
        self.servicer.report(request, None)
        self.assertEqual(len(self.servicer._sync_service._sync_objs_target), 1)
        sync_obj = self.servicer._sync_service._sync_objs_target["test"]
        self.assertEqual(len(sync_obj), 2)

        message = grpc.SyncFinish("test")
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertFalse(response.success)

        self.servicer._sync_service._sync_objs_target["test"] = []
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)

        message = grpc.SyncBarrier("test")
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertFalse(response.success)

        self.servicer._sync_service._finished_barriers = ["test"]
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)

    def test_get_paral_config(self):
        message = grpc.ParallelConfigRequest()
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        self.servicer.report(request, None)
        response = self.servicer.get(request, None)
        config = grpc.deserialize_message(response.data)
        if config:
            self.assertIsInstance(config, grpc.ParallelConfig)

    def test_get_straggler(self):
        message = grpc.StragglerExistRequest()
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        self.servicer.report(request, None)
        response = self.servicer.get(request, None)
        config = grpc.deserialize_message(response.data)
        self.assertIsInstance(config, grpc.NetworkCheckResult)

    def test_check_hardware_reset(self):
        message = grpc.CheckHardwareResetRequest()
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        response = self.servicer.get(request, None)
        config = grpc.deserialize_message(response.data)
        self.assertIsInstance(config, grpc.ParallelConfig)
        self.assertFalse(config.restart)

    def test_join_rendezvous(self):
        request = grpc.JoinRendezvousRequest(
            0, 8, RendezvousName.ELASTIC_TRAINING
        )
        self.servicer._join_rendezvous(request)
        res = self.servicer._num_nodes_waiting(RendezvousName.ELASTIC_TRAINING)
        self.assertEqual(res.waiting_num, 1)
        request = grpc.JoinRendezvousRequest(
            0, 8, RendezvousName.NETWORK_CHECK
        )
        self.servicer._join_rendezvous(request)
        res = self.servicer._num_nodes_waiting(RendezvousName.ELASTIC_TRAINING)
        self.assertEqual(res.waiting_num, 0)

    def test_report_heartbeat(self):
        request = elastic_training_pb2.Message()
        ts = int(time.time())
        message = grpc.HeartBeat(ts)
        request.data = message.serialize()
        request.node_type = NodeType.WORKER
        request.node_id = 0
        self.servicer.report(request, None)
        worker0 = self.servicer._job_manager._job_nodes[NodeType.WORKER][0]
        self.assertEqual(worker0.heartbeat_time, ts)

    def test_sync_checkpoint(self):
        message = grpc.NodeCheckpointState(step=100)
        et_name = RendezvousName.ELASTIC_TRAINING
        rdzv_manager = self.servicer._rdzv_managers[et_name]
        rdzv_manager._latest_rdzv_nodes = [0, 1]
        success = self.servicer._sync_checkpoint(NodeType.WORKER, 0, message)
        self.assertFalse(success)
        success = self.servicer._sync_checkpoint(NodeType.WORKER, 1, message)
        self.assertTrue(success)


class MasterServicerForRayTest(unittest.TestCase):
    def setUp(self) -> None:
        ray.init = mock.MagicMock(return_value=True)
        params = MockRayJobArgs()
        params.initilize()
        speed_monitor = SpeedMonitor()
        self.task_manager = TaskManager(False, speed_monitor)
        self.job_manager = create_job_manager(params, speed_monitor)
        self.job_metric_collector = JobMetricCollector(
            "1", "default", "local", "dlrover"
        )
        self.elastic_ps_service = ElasticPsService()
        self.servicer = MasterServicer(
            task_manager=self.task_manager,
            job_manager=self.job_manager,
            speed_monitor=speed_monitor,
            rdzv_managers={},
            diagnosis_manager=DiagnosisManager(),
            job_metric_collector=self.job_metric_collector,
            elastic_ps_service=self.elastic_ps_service,
        )

    def test_update_node_addr(self):
        request = grpc.NodeMeta()
        task_id = 1
        task_type = NodeType.PS
        addr = "localhost:5001"
        request.type = task_type
        request.id = task_id
        request.addr = "localhost:5001"
        self.job_manager._init_nodes()
        self.servicer._update_node_address(request)
        self.assertEqual(
            self.job_manager._job_nodes[task_type][task_id].service_addr, addr
        )
        for node in self.job_manager._job_nodes[NodeType.PS].values():
            node.status = NodeStatus.RUNNING
        res = self.servicer._query_ps_nodes()
        self.assertEqual(addr, res.nodes[task_id].addr)
        self.assertEqual("", res.nodes[0].addr)

    def test_update_node_event(self):
        request = grpc.NodeEvent(node=grpc.NodeMeta())
        task_id = 1
        task_type = NodeType.PS
        request.node.type = task_type
        request.node.id = task_id
        request.event_type = "Deleted"
        request.message = "OOM"
        self.servicer._update_node_event(request)
        event = ray_event_queue.get()
        event.event_type = "OOM"
