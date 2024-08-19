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

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest import mock

from dlrover.proto import elastic_training_pb2
from dlrover.python.common.constants import (
    DistributionStrategy,
    JobExitReason,
    NodeEventType,
    NodeExitReason,
    NodeStatus,
    NodeType,
    TrainingExceptionLevel,
)
from dlrover.python.common.grpc import (
    DataLoaderConfig,
    GPUStats,
    OptimizerConfig,
    ParallelConfig,
)
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.master.dist_master import DistributedJobMaster
from dlrover.python.master.monitor.error_monitor import SimpleErrorMonitor
from dlrover.python.master.monitor.speed_monitor import SpeedMonitor
from dlrover.python.master.node.dist_job_manager import create_job_manager
from dlrover.python.master.node.event_callback import (
    ClusterContext,
    TaskRescheduleCallback,
    TFPSNodeHandlingCallback,
)
from dlrover.python.master.node.local_job_manager import LocalJobManager
from dlrover.python.master.node.status_flow import (
    NODE_STATE_FLOWS,
    NodeStateFlow,
    get_node_state_flow,
)
from dlrover.python.master.node.training_node import (
    get_critical_worker_index,
    set_critical_node,
    update_nodes_priority,
)
from dlrover.python.master.resource.job import JobResource
from dlrover.python.master.watcher.base_watcher import Node, NodeEvent
from dlrover.python.scheduler.job import LocalJobArgs
from dlrover.python.tests.test_utils import (
    MockK8sAllreduceJobArgs,
    MockK8sPSJobArgs,
    create_task_manager,
    mock_k8s_client,
    new_dataset_splitter,
)

_MOCK_JOB_UUID = "11111"


def get_service_fn(*args):
    return "test:2222"


def _get_node_name(type, id):
    return "{}-{}".format(type, id)


class NodeStatusFlowTest(unittest.TestCase):
    def test_get_node_state_flow(self):
        flow: NodeStateFlow = get_node_state_flow(
            NodeStatus.PENDING, NodeEventType.MODIFIED, NodeStatus.RUNNING
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[4])

        flow = get_node_state_flow(
            NodeStatus.RUNNING, NodeEventType.MODIFIED, NodeStatus.SUCCEEDED
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[7])

        flow = get_node_state_flow(
            NodeStatus.RUNNING, NodeEventType.DELETED, NodeStatus.DELETED
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[10])
        self.assertTrue(flow.should_relaunch)

        flow = get_node_state_flow(
            NodeStatus.SUCCEEDED, NodeEventType.DELETED, NodeStatus.DELETED
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[-2])
        self.assertFalse(flow.should_relaunch)

        flow = get_node_state_flow(
            NodeStatus.PENDING, NodeEventType.DELETED, NodeStatus.DELETED
        )
        self.assertEqual(flow, NODE_STATE_FLOWS[9])
        self.assertTrue(flow.should_relaunch)


class DistributedJobManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()

    def test_job_resource(self):
        job = JobResource()
        job.node_group_resources[NodeType.PS] = NodeGroupResource(
            3, NodeResource(1, 4096)
        )
        job.node_group_resources[NodeType.WORKER] = NodeGroupResource(
            5, NodeResource(1, 4096)
        )
        group_resource = job.get_node_group_resource(NodeType.WORKER)
        self.assertEqual(group_resource.count, 5)
        self.assertEqual(group_resource.node_resource.cpu, 1)
        self.assertEqual(group_resource.node_resource.memory, 4096)
        group_resource = job.get_node_group_resource(NodeType.PS)
        self.assertEqual(group_resource.count, 3)
        self.assertEqual(group_resource.node_resource.cpu, 1)
        self.assertEqual(group_resource.node_resource.memory, 4096)
        node_types = job.get_node_types()
        self.assertListEqual(node_types, [NodeType.PS, NodeType.WORKER])
        self.assertEqual(job.worker_num, 5)
        self.assertEqual(job.ps_num, 3)

        nodes = job.init_job_node_meta(1, get_service_fn, _get_node_name)
        self.assertEqual(len(nodes[NodeType.WORKER]), 5)
        self.assertEqual(len(nodes[NodeType.PS]), 3)
        self.assertEqual(nodes[NodeType.PS][0].id, 0)
        self.assertEqual(nodes[NodeType.PS][0].type, NodeType.PS)
        self.assertEqual(nodes[NodeType.WORKER][2].id, 2)
        self.assertEqual(nodes[NodeType.WORKER][0].type, NodeType.WORKER)
        self.assertEqual(nodes[NodeType.WORKER][0].config_resource.cpu, 1)
        self.assertEqual(
            nodes[NodeType.WORKER][0].config_resource.memory, 4096
        )

    def test_node_priority(self):
        job = JobResource()
        job.node_group_resources[NodeType.PS] = NodeGroupResource(
            3, NodeResource(8, 10240, priority="high")
        )
        job.node_group_resources[NodeType.WORKER] = NodeGroupResource(
            5, NodeResource(8, 10240, priority="0.5")
        )

        nodes = job.init_job_node_meta(1, get_service_fn, _get_node_name)
        update_nodes_priority(nodes)
        ps_priority = []
        for node in nodes[NodeType.PS].values():
            ps_priority.append(node.config_resource.priority)
        self.assertListEqual(ps_priority, ["high"] * 3)
        worker_priority = []
        for node in nodes[NodeType.WORKER].values():
            worker_priority.append(node.config_resource.priority)
        self.assertListEqual(
            worker_priority, ["high", "high", "high", "low", "low"]
        )

    def test_set_critical_node(self):
        job = JobResource()
        job.node_group_resources[NodeType.PS] = NodeGroupResource(
            3, NodeResource(1, 4096)
        )
        job.node_group_resources[NodeType.WORKER] = NodeGroupResource(
            5, NodeResource(1, 4096)
        )

        nodes = job.init_job_node_meta(1, get_service_fn, _get_node_name)
        set_critical_node(
            nodes, ps_relaunch_max_num=2, critical_worker_index={0: 3}
        )
        self.assertTrue(nodes[NodeType.PS][0].critical)
        self.assertEqual(nodes[NodeType.PS][0].max_relaunch_count, 2)
        self.assertTrue(nodes[NodeType.WORKER][0].critical)
        self.assertEqual(nodes[NodeType.WORKER][0].max_relaunch_count, 3)
        self.assertTrue(nodes[NodeType.WORKER][0].critical)
        self.assertFalse(nodes[NodeType.WORKER][1].critical)

    def test_get_critical_worker_index(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        critical_worker = get_critical_worker_index(params)
        self.assertDictEqual(critical_worker, {0: 3})

        params.node_args[NodeType.WORKER].critical_nodes = "0:1"
        critical_worker = get_critical_worker_index(params)
        self.assertDictEqual(critical_worker, {0: 1})

        params.node_args[NodeType.WORKER].critical_nodes = "all"
        critical_worker = get_critical_worker_index(params)
        self.assertDictEqual(critical_worker, {0: 3, 1: 3, 2: 3})

        params.node_args[NodeType.WORKER].critical_nodes = "0"
        critical_worker = get_critical_worker_index(params)
        self.assertDictEqual(critical_worker, {})

    def test_relaunch_node(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        self.assertEqual(manager._ps_relaunch_max_num, 1)
        manager.start()
        self.assertEqual(manager._job_args.job_uuid, _MOCK_JOB_UUID)
        self.assertEqual(len(manager._job_nodes), 4)
        self.assertTrue(manager._job_nodes[NodeType.PS][0].critical)

        node = Node(
            node_type=NodeType.WORKER,
            node_id=1,
            status=NodeStatus.RUNNING,
            config_resource=NodeResource(1, 4096),
            max_relaunch_count=1,
        )

        gpu_stats: list[GPUStats] = [
            GPUStats(
                index=0,
                total_memory_mb=24000,
                used_memory_mb=4000,
                gpu_utilization=55.5,
            )
        ]

        manager.update_node_resource_usage(
            NodeType.WORKER, 0, 0.7, 2048, gpu_stats
        )  # noqa
        self.assertEqual(
            manager._job_nodes[NodeType.WORKER][0].used_resource.cpu, 0.7
        )
        self.assertEqual(
            manager._job_nodes[NodeType.WORKER][0].used_resource.memory, 2048
        )
        self.assertEqual(
            manager._job_nodes[NodeType.WORKER][0].used_resource.gpu_stats,
            gpu_stats,  # noqa
        )

        node_event: NodeEvent = NodeEvent(NodeEventType.MODIFIED, node)
        manager._process_event(node_event)
        self.assertEqual(
            manager._job_nodes[NodeType.WORKER][1].status, NodeStatus.RUNNING
        )
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[5])
        self.assertFalse(should_relaunch)

        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertTrue(should_relaunch)

        node.relaunch_count = node.max_relaunch_count + 1
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertFalse(should_relaunch)

        node.exit_reason = NodeExitReason.FATAL_ERROR
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertFalse(should_relaunch)

        manager.handle_training_failure(
            NodeType.WORKER, 0, level=TrainingExceptionLevel.NODE_ERROR
        )

    def test_get_dead_node_event(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager.start()
        ts = int(time.time())
        manager.collect_node_heart_beat(NodeType.WORKER, 0, ts)
        worker0 = manager._job_nodes[NodeType.WORKER][0]
        self.assertEqual(worker0.heartbeat_time, ts)
        for node in manager._job_nodes[NodeType.WORKER].values():
            node.status = NodeStatus.RUNNING
        events = manager._get_dead_node_event()
        self.assertEqual(len(events), 0)
        for index, node in enumerate(
            manager._job_nodes[NodeType.WORKER].values()
        ):
            node.status = NodeStatus.RUNNING
            now = datetime.now()
            node.heartbeat_time = (now - timedelta(seconds=1000)).timestamp()
            if index == 0:
                node.create_time = now - timedelta(seconds=800)
                node.start_time = now - timedelta(seconds=600)
            else:
                node.create_time = now - timedelta(seconds=1400)
                node.start_time = now - timedelta(seconds=1200)
        events = manager._get_dead_node_event()
        self.assertEqual(len(events), 2)

        nodes_time_info = manager._get_nodes_time_info()
        self.assertIsNotNone(nodes_time_info)
        self.assertEqual(len(nodes_time_info), 3)

    def test_relaunch_training_master(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        group_resources = manager._job_resource.node_group_resources
        group_resources[NodeType.MASTER] = NodeGroupResource(
            1, NodeResource(1, 256)
        )

        manager._init_nodes()
        master = Node(NodeType.MASTER, 0, NodeResource(1, 256))
        manager._job_nodes[NodeType.MASTER][0] = master
        plan = manager._chief_manager.relaunch_node(master)
        self.assertEqual(plan.launch_nodes[0].id, 1)

    def test_process_list_nodes(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()
        for node in manager._job_nodes[NodeType.PS].values():
            node.status = NodeStatus.PENDING
        nodes = []
        for i in range(2):
            node = Node(
                node_type=NodeType.PS,
                node_id=i,
                status=NodeStatus.RUNNING,
                config_resource=NodeResource(1, 4096),
                max_relaunch_count=1,
            )
            nodes.append(node)
        manager._process_list_nodes(nodes)
        ps_ids = list(manager._job_nodes[NodeType.PS].keys())
        self.assertListEqual(ps_ids, [0, 1, 2, 3])

    def test_create_allreduce_job_manager(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        params.distribution_strategy = DistributionStrategy.ALLREDUCE
        params.node_args.pop(NodeType.PS)
        params.node_args.pop(NodeType.CHIEF)
        params.node_args.pop(NodeType.EVALUATOR)
        manager = create_job_manager(params, SpeedMonitor())
        manager._job_optimizer.init_job_resource(manager._job_resource)
        manager._adjust_worker_for_estimator()
        manager._init_nodes()
        manager._init_job_auto_scaler()
        self.assertEqual(len(manager._job_nodes[NodeType.WORKER]), 3)
        manager.start_auto_scaling()
        self.assertEqual(len(manager._job_nodes[NodeType.WORKER]), 3)

    def test_recover_tasks_for_failed_workers(self):
        ds_name_0 = "test-0"
        ds_name_1 = "test-1"
        task_manager = create_task_manager(ds_name_0)
        splitter = new_dataset_splitter(
            False,
            100,
            1000,
            1,
            ds_name_1,
            "table",
        )
        task_manager.new_dataset(
            batch_size=10,
            dataset_size=1000,
            dataset_name=ds_name_1,
            dataset_splitter=splitter,
            task_type=elastic_training_pb2.EVALUATION,
        )

        task_callback = TaskRescheduleCallback(task_manager)
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()
        manager.add_node_event_callback(task_callback)

        dataset_0 = task_manager.get_dataset(ds_name_0)
        dataset_1 = task_manager.get_dataset(ds_name_1)
        task_manager.get_dataset_task(NodeType.WORKER, 0, ds_name_0)
        task_manager.get_dataset_task(NodeType.WORKER, 0, ds_name_1)
        node = Node(
            node_type=NodeType.WORKER,
            node_id=0,
            status=NodeStatus.RUNNING,
            config_resource=NodeResource(1, 4096),
        )
        manager._process_node_events(NODE_STATE_FLOWS[9], node)
        self.assertEqual(len(dataset_0.doing), 0)
        self.assertEqual(len(dataset_1.doing), 0)

    def test_create_initial_nodes(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()
        plan = manager._create_initial_scale_plan()
        self.assertEqual(
            plan.ps_addrs,
            [
                "test-edljob-ps-0.default.svc:2222",
                "test-edljob-ps-1.default.svc:2222",
                "test-edljob-ps-2.default.svc:2222",
            ],
        )
        self.assertEqual(plan.node_group_resources[NodeType.PS].count, 3)
        self.assertEqual(plan.node_group_resources[NodeType.WORKER].count, 3)

    def test_check_worker_status(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()
        self.assertFalse(manager.all_workers_exited())

        for worker in manager._job_nodes[NodeType.WORKER].values():
            worker.status = NodeStatus.FINISHED
        for worker in manager._job_nodes[NodeType.CHIEF].values():
            worker.status = NodeStatus.FINISHED
        for worker in manager._job_nodes[NodeType.EVALUATOR].values():
            worker.status = NodeStatus.FINISHED
        self.assertTrue(manager.all_workers_exited())

        for worker in manager._job_nodes[NodeType.WORKER].values():
            worker.status = NodeStatus.FAILED
        for worker in manager._job_nodes[NodeType.CHIEF].values():
            worker.status = NodeStatus.FAILED
        for worker in manager._job_nodes[NodeType.EVALUATOR].values():
            worker.status = NodeStatus.FAILED
        self.assertTrue(manager.all_workers_failed())

        for worker in manager._job_nodes[NodeType.PS].values():
            worker.status = NodeStatus.FINISHED
        manager._job_nodes[NodeType.WORKER][0].status = NodeStatus.RUNNING
        self.assertFalse(manager.all_critical_node_completed())
        manager._job_nodes[NodeType.WORKER][0].status = NodeStatus.FINISHED
        self.assertTrue(manager.all_critical_node_completed())

    def test_tf_ps_node_handling(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        master = DistributedJobMaster(2222, params)
        master.job_manager._init_nodes()
        master.job_manager._scaler.scale = mock.MagicMock(return_value=True)
        callback = TFPSNodeHandlingCallback(master)

        node = Node(NodeType.PS, 0, None)
        node.config_resource = NodeResource(1, 10240)
        node.exit_reason = NodeExitReason.OOM
        reason = callback.get_job_exit_reason(node)
        self.assertEqual(reason, JobExitReason.PS_OOM_ERROR)

        node.type = NodeType.WORKER
        reason = callback.get_job_exit_reason(node)
        self.assertEqual(reason, JobExitReason.CODE_ERROR)

        master.speed_monitor.add_running_worker(NodeType.WORKER, 0)
        master.speed_monitor.add_running_worker(NodeType.WORKER, 1)
        cluster_context = ClusterContext(master.job_manager)
        master.speed_monitor.set_target_worker_num(2)
        node.exit_reason = NodeExitReason.FATAL_ERROR
        callback.on_node_failed(node, cluster_context)
        self.assertEqual(master.speed_monitor._target_worker_num, 1)
        self.assertEqual(len(master.speed_monitor.running_workers), 1)
        master.speed_monitor.set_target_worker_num(2)
        master.speed_monitor._workers.add(("worker", 0))
        callback.on_node_succeeded(node, cluster_context)
        self.assertEqual(master.speed_monitor._target_worker_num, 1)

    def test_all_running_node_hang(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()

        hang = manager.all_running_node_hanged()
        self.assertFalse(hang)

        for _, nodes in manager._job_nodes.items():
            for _, node in nodes.items():
                node.start_hang_time = time.time() - 3600 * 4
                node.status = NodeStatus.RUNNING
        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.01, 256)
        hang = manager.all_running_node_hanged()
        self.assertTrue(hang)
        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.5, 256)
        hang = manager.all_running_node_hanged()
        self.assertFalse(hang)

    def test_early_stop_part1(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()
        for node in manager._job_nodes[NodeType.PS].values():
            node.status = NodeStatus.PENDING
            node.is_recovered_oom = True
            node.create_time = datetime.now()
        result, reason, msg = manager.should_early_stop()
        self.assertFalse(result)
        self.assertFalse(reason)
        self.assertFalse(msg)

        manager._remove_exited_node = True
        manager._job_nodes[NodeType.WORKER][0].status = NodeStatus.FAILED
        manager.clear_exited_nodes()
        self.assertTrue(manager._job_nodes[NodeType.WORKER][0].is_released)

        for node in manager._job_nodes[NodeType.PS].values():
            node.status = NodeStatus.PENDING
            node.create_time = datetime.now() + timedelta(days=-1)
            node.is_recovered_oom = True
        result, reason, msg = manager.should_early_stop()
        self.assertTrue(result)
        self.assertTrue(reason)
        self.assertTrue(msg)

        for node in manager._job_nodes[NodeType.PS].values():
            node.status = NodeStatus.RUNNING
            node.create_time = datetime.now() + timedelta(days=-1)
            node.is_recovered_oom = True
        result, reason, msg = manager.should_early_stop()
        self.assertFalse(result)
        self.assertFalse(reason)
        self.assertFalse(msg)

    def test_early_stop_part2(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()

        manager.is_all_reduce_type_job = mock.MagicMock(return_value=True)
        manager._worker_manager.is_training_hang_by_pending = mock.MagicMock(
            return_value=True
        )
        result, reason, msg = manager.should_early_stop()
        self.assertTrue(result)
        self.assertEqual(reason, JobExitReason.PENDING_TIMEOUT)
        self.assertTrue(msg)

        manager.is_all_reduce_type_job = mock.MagicMock(return_value=False)
        result, reason, msg = manager.should_early_stop()
        self.assertFalse(result)

    def test_early_stop_part3(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        manager._init_nodes()

        manager.is_all_reduce_type_job = mock.MagicMock(return_value=True)
        manager._worker_manager.is_training_hang_by_insufficient_worker = (
            mock.MagicMock(return_value=True)
        )
        result, reason, msg = manager.should_early_stop()
        self.assertTrue(result)
        self.assertEqual(reason, JobExitReason.UNCOMPLETED_TIMEOUT)
        self.assertTrue(msg)

        manager.is_all_reduce_type_job = mock.MagicMock(return_value=False)
        result, reason, msg = manager.should_early_stop()
        self.assertFalse(result)

    def test_when_node_not_init(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())
        self.assertTrue(not manager._job_nodes)

        manager.update_node_resource_usage(NodeType.WORKER, 0, 10, 10240, None)

    def test_start_and_stop(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, SpeedMonitor())

        manager.start()
        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertIn("node_monitor", active_threads_name)
        self.assertIn("node_heart_beat_monitor", active_threads_name)
        manager.stop()

    def test_concurrency_heart_beat_collecting(self):
        params = MockK8sAllreduceJobArgs()
        worker_size = 10000
        params.initilize(worker_size)
        manager = create_job_manager(params, SpeedMonitor())
        manager.start()

        self.assertEqual(len(manager._job_nodes[NodeType.WORKER]), worker_size)
        for i, node in manager._job_nodes[NodeType.WORKER].items():
            self.assertEqual(node.id, i)
            self.assertEqual(node.heartbeat_time, 0)
        futures = []
        with ThreadPoolExecutor(max_workers=100) as executor:
            for i in range(worker_size):
                futures.append(
                    executor.submit(
                        manager.collect_node_heart_beat, NodeType.WORKER, i, i
                    )
                )

            for future in futures:
                future.result()

        self.assertEqual(len(futures), worker_size)
        for i, node in manager._job_nodes[NodeType.WORKER].items():
            self.assertEqual(node.id, i)
            self.assertEqual(node.heartbeat_time, i)

        manager.stop()


class LocalJobManagerTest(unittest.TestCase):
    def test_local_job_manager(self):
        args = LocalJobArgs("local", "default", "test")
        args.initilize()
        args.node_args[NodeType.WORKER].group_resource.count = 4
        job_mananger = LocalJobManager(
            args, error_monitor=SimpleErrorMonitor()
        )
        job_mananger.start()
        self.assertEqual(len(job_mananger._job_nodes[NodeType.WORKER]), 4)
        gpu_stats: list[GPUStats] = [
            GPUStats(
                index=0,
                total_memory_mb=24000,
                used_memory_mb=4000,
                gpu_utilization=55.5,
            )
        ]
        job_mananger.update_node_resource_usage(
            NodeType.WORKER, 0, 10, 10240, gpu_stats
        )

        worker = job_mananger._job_nodes[NodeType.WORKER][0]
        self.assertEqual(worker.used_resource.cpu, 10)
        self.assertEqual(worker.used_resource.memory, 10240)
        self.assertEqual(worker.used_resource.gpu_stats, gpu_stats)

        dataloader_config = DataLoaderConfig(1, "test_dataloader", 2, 3, 4)
        optimizer_config = OptimizerConfig(1, "test_optimizer", 2)
        paral_config = ParallelConfig(dataloader_config, optimizer_config)
        job_mananger.update_node_paral_config(NodeType.WORKER, 0, paral_config)
        worker = job_mananger._job_nodes[NodeType.WORKER][0]
        self.assertEqual(worker.paral_config, paral_config)
        job_mananger.handle_training_failure(NodeType.WORKER, 3)
