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

import os
import time
import unittest
from unittest import mock

import ray
import requests

from dlrover.proto import elastic_training_pb2
from dlrover.python.common import comm, env_utils
from dlrover.python.common.comm import BaseRequest, GPUStats
from dlrover.python.common.constants import (
    JobStage,
    NodeEventType,
    NodeStatus,
    NodeType,
    PSClusterVersionType,
    RendezvousName,
)
from dlrover.python.common.global_context import Context
from dlrover.python.diagnosis.common.diagnosis_data import WorkerTrainingMetric
from dlrover.python.master.diagnosis.diagnosis_master import DiagnosisMaster
from dlrover.python.master.elastic_training.elastic_ps import ElasticPsService
from dlrover.python.master.elastic_training.rdzv_manager import (
    ElasticTrainingRendezvousManager,
    NetworkCheckRendezvousManager,
)
from dlrover.python.master.elastic_training.sync_service import SyncService
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.dist_job_manager import create_job_manager
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.servicer import (
    GrpcMasterServicer,
    create_master_service,
)
from dlrover.python.master.shard.task_manager import TaskManager
from dlrover.python.master.stats.job_collector import JobMetricCollector
from dlrover.python.tests.test_utils import (
    MockK8sPSJobArgs,
    MockRayJobArgs,
    mock_k8s_client,
)
from dlrover.python.util.queue.queue import RayEventQueue

ray_event_queue = RayEventQueue.singleton_instance()
TEST_SERVER_PORT = 8000


class MasterServicerBasicTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = None

    def tearDown(self) -> None:
        context = Context.singleton_instance()
        context.master_service_type = "grpc"
        if self.server:
            self.server.stop(grace=None)

    def test_http_start_and_stop(self):
        context = Context.singleton_instance()
        context.master_service_type = "http"
        self.server = create_master_service(
            TEST_SERVER_PORT,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.assertIsNotNone(self.server)
        self.assertFalse(self.server.is_serving())

        self.server.start()
        self.assertTrue(self.server.is_serving())

        self.server.stop()
        self.assertFalse(self.server.is_serving())

    def test_grpc_start_and_stop(self):
        self.server = create_master_service(
            TEST_SERVER_PORT,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.assertIsNotNone(self.server)
        self.server.start()
        self.server.stop(grace=None)

    def test_http_basic(self):
        context = Context.singleton_instance()
        context.master_service_type = "http"
        self.server = create_master_service(
            TEST_SERVER_PORT,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        self.server.start()

        response = requests.get("http://localhost:8000/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "Not supported")

        request = BaseRequest()
        request.node_id = 1
        request.node_type = "worker"
        request.data = "test".encode()
        response = requests.post(
            "http://localhost:8000/get", json=request.to_json()
        )
        self.assertEqual(response.status_code, 200)
        response_content = comm.deserialize_message(response.content)
        self.assertIsNotNone(response_content)
        self.assertTrue(response_content.success)

        self.server.stop()


class MasterServicerFunctionalTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        params = MockK8sPSJobArgs()
        params.initilize()
        worker_resource = params.node_args[NodeType.WORKER].group_resource
        worker_resource.node_resource.gpu_num = 1
        worker_resource.node_resource.gpu_type = "a100"
        perf_monitor = PerfMonitor()
        self.task_manager = TaskManager(False, perf_monitor)

        self.job_manager = create_job_manager(params, perf_monitor)
        self.job_context = get_job_context()
        self.job_context.update_job_stage(JobStage.JOB_INIT)

        self.job_manager._init_nodes()
        self.job_manager._init_job_auto_scaler()
        job_nodes = self.job_context.job_nodes_by_type(NodeType.WORKER)
        for node in job_nodes.values():
            node.status = NodeStatus.RUNNING
            self.job_context.update_job_node(node)
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
        self.servicer = GrpcMasterServicer(
            task_manager=self.task_manager,
            job_manager=self.job_manager,
            perf_monitor=perf_monitor,
            rdzv_managers=rdzv_managers,
            diagnosis_manager=DiagnosisMaster(),
            job_metric_collector=self.job_metric_collector,
            elastic_ps_service=self.elastic_ps_service,
            sync_service=sync_service,
        )

    def tearDown(self) -> None:
        os.environ.clear()
        self.job_context.clear_job_nodes()

    def test_query_running_nodes(self):
        request = elastic_training_pb2.Message()
        message = comm.RunningNodesRequest()
        request.data = message.serialize()
        res = self.servicer.get(request, None)
        ret: comm.RunningNodes = comm.deserialize_message(res.data)
        self.assertEqual(len(ret.nodes), 3)

    def test_dataset_service(self):
        request = comm.DatasetShardParams()
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

        request = comm.TaskRequest("test")
        task: comm.Task = self.servicer._get_task(NodeType.WORKER, 0, request)
        self.assertEqual(task.task_id, 0)
        self.assertEqual(task.shard.start, 0)
        self.assertEqual(task.shard.end, 100)

        request = comm.TaskResult(dataset_name="test", task_id=0)
        request.task_id = 0
        request.dataset_name = "test"
        self.servicer._report_task_result(request)
        self.assertEqual(len(self.task_manager._datasets["test"].doing), 0)

        request = comm.ShardCheckpointRequest("test")
        request.dataset_name = "test"

        checkpoint = self.servicer._get_shard_checkpoint(request)
        self.assertLessEqual(10, len(checkpoint.content))
        self.servicer._restore_shard_checkpoint(checkpoint)

    def test_metric_service(self):
        self.job_manager._init_nodes()
        self.job_manager._init_job_auto_scaler()
        request = comm.ResourceStats(gpu_stats=[])
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
            gpu_stats_message = comm.GPUStats()
            gpu_stats_message.index = gpu.index
            gpu_stats_message.total_memory_mb = gpu.total_memory_mb
            gpu_stats_message.used_memory_mb = gpu.used_memory_mb
            gpu_stats_message.gpu_utilization = gpu.gpu_utilization
            request.gpu_stats.append(gpu_stats_message)
        self.servicer._update_node_resource_usage(NodeType.WORKER, 0, request)
        self.servicer._update_node_resource_usage(NodeType.PS, 0, request)

        request = comm.ModelInfo(
            tensor_stats=comm.TensorStats(), op_stats=comm.OpStats()
        )
        request.tensor_stats.variable_count = 100
        request.tensor_stats.total_variable_size = 10000

        request.op_stats.op_count = 100
        request.op_stats.flops = 10000
        self.servicer._collect_model_info(request)
        reporter = self.job_metric_collector._stats_reporter
        reporter._runtime_stats = []
        self.assertEqual(reporter._model_info.op_stats.flops, 10000)

        job_nodes = self.job_context.job_nodes()
        worker0 = job_nodes[NodeType.WORKER][0]
        worker0.status = NodeStatus.RUNNING
        self.job_context.update_job_node(worker0)

        ps0 = job_nodes[NodeType.PS][0]
        ps0.status = NodeStatus.RUNNING
        self.job_context.update_job_node(ps0)

        request = comm.GlobalStep()
        self.task_manager._perf_monitor.add_running_worker(NodeType.WORKER, 0)
        self.task_manager._perf_monitor.set_target_worker_num(1)
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
        nodes = self.job_context.job_nodes_by_type(NodeType.PS)
        for node in nodes.values():
            node.status = NodeStatus.RUNNING
            self.job_context.update_job_node(node)
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
        message = comm.ClusterVersionRequest(NodeType.WORKER, 0, "local")
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        response = self.servicer.get(request, None)
        res_msg = comm.deserialize_message(response.data)
        self.assertEqual(res_msg.version, 0)

        message = comm.ClusterVersionRequest(NodeType.PS, 0, "local")
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        response = self.servicer.get(request, None)
        res_msg = comm.deserialize_message(response.data)
        self.assertEqual(res_msg.version, 0)

    def test_get_training_status(self):
        message = comm.TrainingStatusRequest()
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        response = self.servicer.get(request, None)
        res_msg: comm.TrainingStatus = comm.deserialize_message(response.data)
        self.assertEqual(res_msg.status, 3)

    def test_num_nodes_waiting(self):
        message = comm.WaitingNodeNumRequest(
            0, 8, RendezvousName.ELASTIC_TRAINING
        )
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        self.servicer._rdzv_managers[
            RendezvousName.ELASTIC_TRAINING
        ]._waiting_nodes = {0: 8}
        response = self.servicer.get(request, None)
        res_msg: comm.RendezvousState = comm.deserialize_message(response.data)
        self.assertEqual(res_msg.waiting_num, 1)

    def test_report(self):
        request = elastic_training_pb2.Message()
        request.data = b""
        response = self.servicer.report(request, None)
        self.assertFalse(response.success)

    def test_report_task_result(self):
        request = elastic_training_pb2.Message()
        message = comm.TaskResult("test", 0, "error")
        dataset_params = comm.DatasetShardParams(
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

        message = comm.TaskResult("test", 0, "")
        request.data = message.serialize()
        self.servicer._start_autoscale = False
        self.servicer._perf_monitor.completed_global_step == 0
        self.servicer._start_training_time = time.time() - 3600
        response = self.servicer.report(request, None)
        self.assertTrue(self.servicer._start_autoscale)
        self.assertTrue(response.success)

    def test_update_cluster_version(self):
        request = elastic_training_pb2.Message()
        message = comm.ClusterVersion(
            NodeType.WORKER, 0, PSClusterVersionType.LOCAL, 1
        )
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)
        self.assertEqual(
            self.servicer._elastic_ps_service._worker_local_version[0], 1
        )

        message = comm.ClusterVersion(
            NodeType.WORKER, 0, PSClusterVersionType.RESTORED, 1
        )
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)
        self.assertEqual(
            self.servicer._elastic_ps_service._worker_restored_version[0], 1
        )

        message = comm.ClusterVersion(
            NodeType.PS, 0, PSClusterVersionType.GLOBAL, 1
        )
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)
        self.assertEqual(self.servicer._elastic_ps_service._global_version, 1)

    def test_sync(self):
        request = elastic_training_pb2.Message()
        message = comm.SyncJoin("test")
        request.data = message.serialize()
        request.node_type = NodeType.WORKER
        request.node_id = 0
        self.servicer.report(request, None)
        self.assertEqual(len(self.servicer._sync_service._sync_objs_target), 1)
        sync_obj = self.servicer._sync_service._sync_objs_target["test"]
        self.assertEqual(len(sync_obj), 2)

        message = comm.SyncFinish("test")
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertFalse(response.success)

        self.servicer._sync_service._sync_objs_target["test"] = []
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)

        message = comm.SyncBarrier("test")
        request.data = message.serialize()
        response = self.servicer.report(request, None)
        self.assertFalse(response.success)

        self.servicer._sync_service._finished_barriers = ["test"]
        response = self.servicer.report(request, None)
        self.assertTrue(response.success)

    def test_get_paral_config(self):
        message = comm.ParallelConfigRequest()
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        self.servicer.report(request, None)
        response = self.servicer.get(request, None)
        config = comm.deserialize_message(response.data)
        if config:
            self.assertIsInstance(config, comm.ParallelConfig)

    def test_get_straggler(self):
        message = comm.StragglerExistRequest()
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        self.servicer.report(request, None)
        response = self.servicer.get(request, None)
        config = comm.deserialize_message(response.data)
        self.assertIsInstance(config, comm.NetworkCheckResult)

    def test_check_hardware_reset(self):
        message = comm.CheckHardwareResetRequest()
        request = elastic_training_pb2.Message()
        request.data = message.serialize()
        response = self.servicer.get(request, None)
        config = comm.deserialize_message(response.data)
        self.assertIsInstance(config, comm.ParallelConfig)
        self.assertFalse(config.restart)

    def test_join_rendezvous(self):
        request = comm.JoinRendezvousRequest(
            0, 8, RendezvousName.ELASTIC_TRAINING
        )
        self.servicer._join_rendezvous(request)
        res = self.servicer._num_nodes_waiting(RendezvousName.ELASTIC_TRAINING)
        self.assertEqual(res.waiting_num, 1)
        request = comm.JoinRendezvousRequest(
            0, 8, RendezvousName.NETWORK_CHECK
        )
        self.servicer._join_rendezvous(request)
        res = self.servicer._num_nodes_waiting(RendezvousName.ELASTIC_TRAINING)
        self.assertEqual(res.waiting_num, 0)

    def test_report_heartbeat(self):
        request = elastic_training_pb2.Message()
        ts = int(time.time())
        message = comm.HeartBeat(ts)
        request.data = message.serialize()
        request.node_type = NodeType.WORKER
        request.node_id = 0
        self.servicer.get(request, None)

        worker0 = self.job_context.job_node(NodeType.WORKER, 0)
        self.assertEqual(worker0.heartbeat_time, ts)

    def test_sync_checkpoint(self):
        message = comm.NodeCheckpointState(step=100)
        et_name = RendezvousName.ELASTIC_TRAINING
        rdzv_manager = self.servicer._rdzv_managers[et_name]
        rdzv_manager._latest_rdzv_nodes = [0, 1]
        success = self.servicer._sync_checkpoint(NodeType.WORKER, 0, message)
        self.assertFalse(success)
        success = self.servicer._sync_checkpoint(NodeType.WORKER, 1, message)
        self.assertTrue(success)

    def test_report_node_diagnosis_data(self):
        test = WorkerTrainingMetric(
            data_content="test123",
            node_id=env_utils.get_node_id(),
            node_type=env_utils.get_node_type(),
            node_rank=env_utils.get_node_rank(),
            is_final_result=True,
        )

        request = comm.DiagnosisReportData(
            test.__class__.__name__,
            test.to_json(),
            test.node_rank,
        )
        self.assertTrue(self.servicer._report_node_diagnosis_data(request))

    def test_deal_with_reported_node_event(self):
        request = comm.NodeEvent(node=comm.NodeMeta())
        task_id = 1
        task_type = NodeType.PS
        request.node.type = task_type
        request.node.id = task_id
        request.event_type = NodeEventType.DELETED
        request.message = "OOM"
        self.assertTrue(self.servicer._deal_with_reported_node_event(request))
        self.assertFalse(
            self.job_manager._job_context.job_node(
                task_type, task_id
            ).is_succeeded_and_exited()
        )

        request.event_type = NodeEventType.NODE_CHECK_FAILED
        request.message = ""
        self.assertTrue(self.servicer._deal_with_reported_node_event(request))
        self.assertTrue(
            self.job_manager._job_context.job_node(
                task_type, task_id
            ).is_node_check_failed()
        )

        request.event_type = NodeEventType.SUCCEEDED_EXITED
        request.message = ""
        self.assertTrue(self.servicer._deal_with_reported_node_event(request))
        self.assertTrue(
            self.job_manager._job_context.job_node(
                task_type, task_id
            ).is_succeeded_and_exited()
        )
        self.assertFalse(
            self.job_manager._job_context.job_node(
                task_type, task_id
            ).is_node_check_failed()
        )

        task_id = 2
        request.event_type = NodeEventType.FAILED_EXITED
        request.node.id = task_id
        request.message = ""
        self.assertTrue(self.servicer._deal_with_reported_node_event(request))
        self.assertTrue(
            self.job_manager._job_context.job_node(
                task_type, task_id
            ).is_failed_and_exited()
        )
        self.assertFalse(
            self.job_manager._job_context.job_node(
                task_type, task_id
            ).is_node_check_failed()
        )


class MasterServicerForRayTest(unittest.TestCase):
    def setUp(self) -> None:
        ray.init = mock.MagicMock(return_value=True)
        params = MockRayJobArgs()
        params.initilize()
        perf_monitor = PerfMonitor()
        self.task_manager = TaskManager(False, perf_monitor)
        self.job_manager = create_job_manager(params, perf_monitor)
        self.job_metric_collector = JobMetricCollector(
            "1", "default", "local", "dlrover"
        )
        self.elastic_ps_service = ElasticPsService()
        self.servicer = GrpcMasterServicer(
            task_manager=self.task_manager,
            job_manager=self.job_manager,
            perf_monitor=perf_monitor,
            rdzv_managers={},
            diagnosis_manager=DiagnosisMaster(),
            job_metric_collector=self.job_metric_collector,
            elastic_ps_service=self.elastic_ps_service,
        )
        self.job_context = get_job_context()

    def tearDown(self) -> None:
        self.job_context.clear_job_nodes()

    def test_update_node_addr(self):
        request = comm.NodeMeta()
        task_id = 1
        task_type = NodeType.PS
        addr = "localhost:5001"
        request.type = task_type
        request.id = task_id
        request.addr = "localhost:5001"
        self.job_manager._init_nodes()
        self.servicer._update_node_address(request)
        node = self.job_context.job_node(task_type, task_id)
        self.assertEqual(node.service_addr, addr)
        ps_nodes = self.job_context.job_nodes_by_type(NodeType.PS)
        for node in ps_nodes.values():
            node.status = NodeStatus.RUNNING
            self.job_context.update_job_node(node)
        res = self.servicer._query_ps_nodes()
        self.assertEqual(addr, res.nodes[task_id].addr)
        self.assertEqual("", res.nodes[0].addr)
