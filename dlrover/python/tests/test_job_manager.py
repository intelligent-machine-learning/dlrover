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
from typing import Dict
from unittest import mock
from unittest.mock import MagicMock, patch

from kubernetes import client

from dlrover.proto import elastic_training_pb2
from dlrover.python.common.comm import (
    DataLoaderConfig,
    GPUStats,
    OptimizerConfig,
    ParallelConfig,
)
from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    JobExitReason,
    JobStage,
    NodeEventType,
    NodeExitReason,
    NodeStatus,
    NodeType,
    PreCheckStatus,
    TrainingExceptionLevel,
)
from dlrover.python.common.node import (
    NodeEvent,
    NodeGroupResource,
    NodeResource,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NoAction,
    NodeAction,
)
from dlrover.python.master.dist_master import DistributedJobMaster
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.dist_job_manager import (
    DistributedJobManager,
    create_job_manager,
)
from dlrover.python.master.node.event_callback import (
    ClusterContext,
    TaskRescheduleCallback,
    TFPSNodeHandlingCallback,
)
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.node.local_job_manager import LocalJobManager
from dlrover.python.master.node.status_flow import (
    ALLOWED_TRANSITIONS,
    NODE_STATE_FLOWS,
    NodeStateFlow,
    get_node_state_flow,
)
from dlrover.python.master.node.training_node import (
    _dlrover_context,
    get_critical_worker_index,
    get_pending_timeout,
    set_critical_node,
    update_nodes_priority,
)
from dlrover.python.master.resource.job import JobResource
from dlrover.python.master.watcher.base_watcher import Node
from dlrover.python.scheduler.job import LocalJobArgs
from dlrover.python.tests.test_utils import (
    MockK8sAllreduceJobArgs,
    MockK8sJobWithoutCPURequestArgs,
    MockK8sPSJobArgs,
    create_pod,
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

    def test_allowed_transitions(self):
        self.assertTrue(
            NodeStatus.RUNNING in ALLOWED_TRANSITIONS[NodeStatus.RUNNING]
        )
        self.assertFalse(
            NodeStatus.PENDING in ALLOWED_TRANSITIONS[NodeStatus.RUNNING]
        )


class DistributedJobManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        self.job_context = get_job_context()

    def tearDown(self):
        self.job_context.clear_job_nodes()
        self.job_context._request_stop = False

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
        manager = create_job_manager(params, PerfMonitor())
        self.assertEqual(manager._ps_relaunch_max_num, 1)
        manager.start()

        manager._job_optimizer.adjust_oom_resource = MagicMock(
            return_value=None
        )

        # reset failed nodes for testing
        self.job_context._failed_nodes = {}
        self.assertEqual(manager._job_args.job_uuid, _MOCK_JOB_UUID)

        job_nodes = self.job_context.job_nodes()
        self.assertEqual(len(job_nodes), 4)
        self.assertTrue(job_nodes[NodeType.PS][0].critical)

        node = Node(
            node_type=NodeType.WORKER,
            node_id=1,
            status=NodeStatus.RUNNING,
            config_resource=NodeResource(1, 4096),
            max_relaunch_count=1,
            node_group=1000,
            node_group_size=1,
            node_group_id="rack-0",
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
        job_nodes = self.job_context.job_nodes()
        self.assertEqual(job_nodes[NodeType.WORKER][0].used_resource.cpu, 0.7)
        self.assertEqual(
            job_nodes[NodeType.WORKER][0].used_resource.memory, 2048
        )
        self.assertEqual(
            job_nodes[NodeType.WORKER][0].used_resource.gpu_stats,
            gpu_stats,  # noqa
        )

        node_event: NodeEvent = NodeEvent(NodeEventType.MODIFIED, node)
        manager._process_event(node_event)
        job_nodes = self.job_context.job_nodes()
        self.assertEqual(
            job_nodes[NodeType.WORKER][1].status, NodeStatus.RUNNING
        )
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[5])
        self.assertFalse(should_relaunch)

        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertTrue(should_relaunch)

        self.job_context.update_job_stage(JobStage.JOB_STOPPING)
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertFalse(should_relaunch)
        self.job_context.update_job_stage(JobStage.JOB_INIT)

        node.relaunch_count = node.max_relaunch_count + 1
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertFalse(should_relaunch)

        node.exit_reason = NodeExitReason.FATAL_ERROR
        should_relaunch = manager._should_relaunch(node, NODE_STATE_FLOWS[6])
        self.assertFalse(should_relaunch)

        self.assertEqual(self.job_context.get_failed_node_cnt(), 0)
        manager.handle_training_failure(
            NodeType.WORKER, 0, level=TrainingExceptionLevel.NODE_ERROR
        )
        manager.handle_training_failure(
            NodeType.WORKER, 0, level=TrainingExceptionLevel.NODE_ERROR
        )
        self.assertEqual(self.job_context.get_failed_node_cnt(), 1)
        manager.handle_training_failure(
            NodeType.WORKER,
            1,
            level=TrainingExceptionLevel.NODE_ERROR,
            error_data="test_reason",
        )
        self.assertEqual(self.job_context.get_failed_node_cnt(), 2)

        # reset relaunch count
        node.relaunch_count = 0
        node.exit_reason = NodeExitReason.OOM
        node.config_resource.memory = 655
        self.assertTrue(manager._should_relaunch(node, NODE_STATE_FLOWS[6]))

        node.config_resource.memory = 65537
        self.assertFalse(manager._should_relaunch(node, NODE_STATE_FLOWS[6]))

        node.config_resource.memory = 655
        manager.is_all_reduce_type_job = MagicMock(return_value=True)
        node.exit_reason = NodeExitReason.OOM
        self.assertFalse(manager._should_relaunch(node, NODE_STATE_FLOWS[6]))

        node.exit_reason = NodeExitReason.RELAUNCHED
        self.assertFalse(manager._should_relaunch(node, NODE_STATE_FLOWS[6]))

    def test_relaunch_node_group(self):
        params = MockK8sAllreduceJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()
        manager._scaler.scale = mock.MagicMock(return_value=None)

        manager._max_group_relaunch_count = -1
        self.job_context.clear_job_node_groups()
        node = Node(
            NodeType.WORKER,
            0,
            rank_index=0,
            status=NodeStatus.PENDING,
            node_group=0,
            node_group_size=1,
            relaunchable=True,
        )
        self.job_context.update_job_node_by_group(node)
        self.assertFalse(manager._should_relaunch_node_group(0))
        manager._max_group_relaunch_count = 3

        self.job_context.clear_job_node_groups()
        node = Node(
            NodeType.WORKER,
            0,
            rank_index=0,
            status=NodeStatus.PENDING,
            node_group=0,
            node_group_size=1,
            relaunchable=True,
        )
        self.job_context.update_job_node_by_group(node)
        self.assertTrue(manager._should_relaunch_node_group(0))

        self.job_context.clear_job_node_groups()
        node = Node(
            NodeType.WORKER,
            0,
            rank_index=0,
            status=NodeStatus.PENDING,
            node_group=0,
            node_group_size=1,
            relaunchable=True,
        )
        self.job_context.update_job_node_by_group(node)
        manager._relaunched_groups.append(0)
        self.assertFalse(manager._should_relaunch_node_group(0))
        manager._relaunched_groups.clear()

        node = Node(
            NodeType.WORKER,
            1,
            rank_index=1,
            status=NodeStatus.PENDING,
            node_group=0,
            node_group_size=1,
            relaunchable=False,
        )
        self.job_context.update_job_node_by_group(node)
        self.assertFalse(manager._should_relaunch_node_group(0))

        self.job_context.clear_job_node_groups()
        self.job_context.clear_job_nodes()

        node0 = Node(
            NodeType.WORKER,
            0,
            rank_index=0,
            status=NodeStatus.RUNNING,
            node_group=0,
            node_group_size=1,
            node_group_id="rack0",
            relaunchable=True,
            name="test-0",
        )
        node1 = Node(
            NodeType.WORKER,
            1,
            rank_index=1,
            status=NodeStatus.PENDING,
            node_group=0,
            node_group_size=1,
            node_group_id="rack0",
            relaunchable=True,
            name="test-1",
        )
        self.job_context.update_job_node(node0)
        self.job_context.update_job_node(node1)
        self.job_context.update_job_node_by_group(node0)
        self.job_context.update_job_node_by_group(node1)
        self.assertTrue(manager._should_relaunch_node_group(0))

        self.job_context.update_job_stage(JobStage.JOB_STOPPING)
        self.assertFalse(manager._should_relaunch_node_group(0))
        self.job_context.update_job_stage(JobStage.JOB_RUNNING)

        plan = manager._relaunch_node_group(0)
        self.assertEqual(manager._relaunched_groups, [0])
        self.assertEqual(plan.launch_nodes[0].id, 2)
        self.assertEqual(plan.launch_nodes[0].rank_index, 0)
        self.assertEqual(plan.launch_nodes[0].group, 1001)
        self.assertEqual(plan.launch_nodes[0].group_size, 1)
        self.assertEqual(plan.launch_nodes[0].group_id, "")

        self.assertEqual(plan.launch_nodes[1].id, 3)
        self.assertEqual(plan.launch_nodes[1].rank_index, 1)
        self.assertEqual(plan.launch_nodes[1].group, 1001)
        self.assertEqual(plan.launch_nodes[1].group_size, 1)
        self.assertEqual(plan.launch_nodes[1].group_id, "")

        self.assertEqual(plan.remove_nodes[0].id, 0)
        self.assertEqual(plan.remove_nodes[0].rank_index, 0)
        self.assertEqual(plan.remove_nodes[0].group, 0)
        self.assertEqual(plan.remove_nodes[0].group_size, 1)
        self.assertEqual(plan.remove_nodes[0].group_id, "rack0")

        self.assertEqual(plan.remove_nodes[1].id, 1)
        self.assertEqual(plan.remove_nodes[1].rank_index, 1)
        self.assertEqual(plan.remove_nodes[1].group, 0)
        self.assertEqual(plan.remove_nodes[1].group_size, 1)
        self.assertEqual(plan.remove_nodes[1].group_id, "rack0")

        self.job_context.clear_job_node_groups()
        self.job_context.clear_job_nodes()

    def test_relaunch_under_deleted_event(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager.start()

        pods = []
        for i in range(2):
            labels = {
                ElasticJobLabel.APP_NAME: "test",
                ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.WORKER,
                ElasticJobLabel.REPLICA_INDEX_KEY: str(i),
                ElasticJobLabel.RANK_INDEX_KEY: str(i),
                ElasticJobLabel.RELAUNCH_COUNT: "0",
            }
            pod = create_pod(labels)
            pods.append(pod)

        return_pods = client.V1PodList(
            items=pods, metadata=client.V1ListMeta(resource_version="12345678")
        )
        manager._k8s_client.list_namespaced_pod = mock.MagicMock(
            return_value=return_pods
        )

        # deleted event + running status
        node0 = Node(
            NodeType.WORKER,
            0,
            NodeResource(0, 0),
            rank_index=0,
            status=NodeStatus.RUNNING,
        )
        manager._process_event(NodeEvent(NodeEventType.DELETED, node0))

        # modified event + deleted status
        node0 = Node(
            NodeType.WORKER,
            0,
            NodeResource(0, 0),
            rank_index=0,
            status=NodeStatus.DELETED,
            node_group=100,
            node_group_size=1,
            node_group_id="rack0",
        )
        manager._process_event(NodeEvent(NodeEventType.MODIFIED, node0))

        # modified event + running status
        node0 = Node(
            NodeType.WORKER,
            0,
            NodeResource(0, 0),
            rank_index=0,
            status=NodeStatus.RUNNING,
            node_group=100,
            node_group_size=1,
            node_group_id="rack0",
        )
        manager._process_event(NodeEvent(NodeEventType.MODIFIED, node0))

        # modified event + running status + no-heartbeat reason
        node0 = Node(
            NodeType.WORKER,
            0,
            NodeResource(0, 0),
            rank_index=0,
            status=NodeStatus.RUNNING,
            node_group=100,
            node_group_size=1,
            node_group_id="rack0",
        )
        node0.exit_reason = NodeExitReason.NO_HEARTBEAT
        manager._process_event(NodeEvent(NodeEventType.MODIFIED, node0))

    def test_get_dead_node_event(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()
        ts = int(time.time())
        manager.collect_node_heart_beat(NodeType.WORKER, 0, ts)

        job_nodes = self.job_context.job_nodes()
        worker0 = job_nodes[NodeType.WORKER][0]
        self.assertEqual(worker0.heartbeat_time, ts)
        for node in job_nodes[NodeType.WORKER].values():
            node.status = NodeStatus.RUNNING
            self.job_context.update_job_node(node)
        events = manager._get_dead_node_event()
        self.assertEqual(len(events), 0)

        job_nodes = self.job_context.job_nodes()
        for index, node in enumerate(job_nodes[NodeType.WORKER].values()):
            node.status = NodeStatus.RUNNING
            now = datetime.now()
            node.heartbeat_time = (now - timedelta(seconds=1000)).timestamp()
            if index == 0:
                node.create_time = now - timedelta(seconds=800)
                node.start_time = now - timedelta(seconds=500)
            else:
                node.create_time = now - timedelta(seconds=1400)
                node.start_time = now - timedelta(seconds=1200)
            self.job_context.update_job_node(node)
        events = manager._get_dead_node_event()
        self.assertEqual(len(events), 2)

        nodes_time_info = manager._get_nodes_time_info()
        self.assertIsNotNone(nodes_time_info)
        self.assertEqual(len(nodes_time_info), 3)

        job_nodes = self.job_context.job_nodes()
        for index, node in enumerate(job_nodes[NodeType.WORKER].values()):
            node.status = NodeStatus.RUNNING
            now = datetime.now()
            node.heartbeat_time = (now - timedelta(seconds=1000)).timestamp()
            if index == 0:
                node.create_time = now - timedelta(seconds=800)
                node.start_time = now - timedelta(seconds=600)
            else:
                if index == 1:
                    node.reported_status = (NodeEventType.SUCCEEDED_EXITED, 0)
                node.create_time = now - timedelta(seconds=1400)
                node.start_time = now - timedelta(seconds=1200)
            self.job_context.update_job_node(node)
        events = manager._get_dead_node_event()
        self.assertEqual(len(events), 1)

        job_nodes = self.job_context.job_nodes()
        for index, node in enumerate(job_nodes[NodeType.WORKER].values()):
            node.status = NodeStatus.RUNNING
            now = datetime.now()
            node.heartbeat_time = (now - timedelta(seconds=1000)).timestamp()
            node.reported_status = (NodeEventType.FAILED_EXITED, 0)
            node.create_time = now - timedelta(seconds=800)
            node.start_time = now - timedelta(seconds=600)
            self.job_context.update_job_node(node)
        events = manager._get_dead_node_event()
        self.assertEqual(len(events), 0)

        job_nodes = self.job_context.job_nodes()
        for index, node in enumerate(job_nodes[NodeType.WORKER].values()):
            node.status = NodeStatus.RUNNING
            now = datetime.now()
            node.heartbeat_time = (now - timedelta(seconds=1000)).timestamp()
            if index == 0:
                node.reported_status = (NodeEventType.SUCCEEDED_EXITED, 0)
            else:
                node.reported_status = (NodeEventType.FAILED_EXITED, 0)
            node.create_time = now - timedelta(seconds=1400)
            node.start_time = now - timedelta(seconds=1200)
            self.job_context.update_job_node(node)
        events = manager._get_dead_node_event()
        self.assertEqual(len(events), 2)

    def test_relaunch_training_master(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        group_resources = manager._job_resource.node_group_resources
        group_resources[NodeType.MASTER] = NodeGroupResource(
            1, NodeResource(1, 256)
        )

        manager._init_nodes()
        master = Node(NodeType.MASTER, 0, NodeResource(1, 256))
        self.job_context.update_job_node(master)
        plan = manager._chief_manager.relaunch_node(master)
        self.assertEqual(plan.launch_nodes[0].id, 1)

    def test_process_list_nodes(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()
        job_nodes = self.job_context.job_nodes()
        self.assertFalse(4 in job_nodes[NodeType.WORKER])
        for node in job_nodes[NodeType.PS].values():
            node.status = NodeStatus.PENDING
            self.job_context.update_job_node(node)
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
        nodes.append(
            Node(
                node_type=NodeType.WORKER,
                node_id=4,
                status=NodeStatus.RUNNING,
                config_resource=NodeResource(1, 4096),
                max_relaunch_count=1,
            )
        )
        manager._process_list_nodes(nodes)

        job_nodes = self.job_context.job_nodes()
        ps_ids = list(job_nodes[NodeType.PS].keys())
        self.assertListEqual(ps_ids, [0, 1, 2])
        self.assertTrue(4 in self.job_context.job_nodes()[NodeType.WORKER])

        self.assertIsNone(
            self.job_context.job_nodes()[NodeType.WORKER][4].group
        )
        self.assertIsNone(
            self.job_context.job_nodes()[NodeType.WORKER][4].group_size
        )
        self.assertIsNone(
            self.job_context.job_nodes()[NodeType.WORKER][4].group_id
        )
        new_node = Node(
            node_type=NodeType.WORKER,
            node_id=100,
            rank_index=100,
            status=NodeStatus.RUNNING,
            config_resource=NodeResource(1, 4096),
            max_relaunch_count=1,
            node_group=1024,
            node_group_size=1,
            node_group_id="rack-0",
        )
        manager._process_list_nodes([new_node])
        self.assertEqual(
            self.job_context.job_nodes()[NodeType.WORKER][100].group, 1024
        )
        self.assertEqual(
            self.job_context.job_nodes()[NodeType.WORKER][100].group_size, 1
        )
        self.assertEqual(
            self.job_context.job_nodes()[NodeType.WORKER][100].group_id,
            "rack-0",
        )
        self.assertEqual(self.job_context.job_node_groups()[1024][100].id, 100)
        self.assertEqual(
            self.job_context.job_node_groups()[1024][100].rank_index, 100
        )
        self.assertEqual(
            self.job_context.job_node_groups()[1024][100].group, 1024
        )
        self.assertEqual(
            self.job_context.job_node_groups()[1024][100].group_size, 1
        )

    @patch.object(DistributedJobManager, "_process_event")
    def test_process_list_nodes_for_empty_case(self, mock_method):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        job_nodes = {
            NodeType.PS: {
                0: Node(
                    node_type=NodeType.PS,
                    node_id=0,
                    status=NodeStatus.RUNNING,
                    config_resource=NodeResource(1, 4096),
                    max_relaunch_count=1,
                )
            },
            NodeType.WORKER: {
                1: Node(
                    node_type=NodeType.WORKER,
                    node_id=1,
                    status=NodeStatus.RUNNING,
                    config_resource=NodeResource(1, 4096),
                    max_relaunch_count=1,
                )
            },
        }
        self.job_context.update_job_nodes(job_nodes)
        manager._process_list_nodes([])
        self.assertEqual(mock_method.call_count, 2)

    def test_create_allreduce_job_manager(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        params.distribution_strategy = DistributionStrategy.ALLREDUCE
        params.node_args.pop(NodeType.PS)
        params.node_args.pop(NodeType.CHIEF)
        params.node_args.pop(NodeType.EVALUATOR)
        manager = create_job_manager(params, PerfMonitor())
        manager._job_optimizer.init_job_resource(manager._job_resource)
        manager._adjust_worker_for_estimator()
        manager._init_nodes()
        manager._init_job_auto_scaler()

        job_nodes = self.job_context.job_nodes()
        self.assertEqual(len(job_nodes[NodeType.WORKER]), 3)
        manager.start_auto_scaling()
        job_nodes = self.job_context.job_nodes()
        self.assertEqual(len(job_nodes[NodeType.WORKER]), 3)

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
        manager = create_job_manager(params, PerfMonitor())
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
        manager = create_job_manager(params, PerfMonitor())
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
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()
        self.assertFalse(manager.all_workers_exited())

        job_nodes = self.job_context.job_nodes()

        for worker in job_nodes[NodeType.WORKER].values():
            worker.status = NodeStatus.FINISHED
        for worker in job_nodes[NodeType.CHIEF].values():
            worker.status = NodeStatus.FINISHED
        for worker in job_nodes[NodeType.EVALUATOR].values():
            worker.status = NodeStatus.FINISHED
        self.job_context.update_job_nodes(job_nodes)
        self.assertTrue(manager.all_workers_exited())

        job_nodes = self.job_context.job_nodes()
        for worker in job_nodes[NodeType.WORKER].values():
            worker.status = NodeStatus.FAILED
            self.job_context.update_job_node(worker)
        for worker in job_nodes[NodeType.CHIEF].values():
            worker.status = NodeStatus.FAILED
            self.job_context.update_job_node(worker)
        for worker in job_nodes[NodeType.EVALUATOR].values():
            worker.status = NodeStatus.FAILED
            self.job_context.update_job_node(worker)
        self.assertTrue(manager.all_workers_failed())

        job_nodes = self.job_context.job_nodes()
        for worker in job_nodes[NodeType.PS].values():
            worker.status = NodeStatus.FINISHED
            self.job_context.update_job_node(worker)
        job_nodes[NodeType.WORKER][0].status = NodeStatus.RUNNING
        self.job_context.update_job_node(job_nodes[NodeType.WORKER][0])
        self.assertFalse(manager.all_critical_node_completed())
        job_nodes[NodeType.WORKER][0].status = NodeStatus.FINISHED
        self.job_context.update_job_node(job_nodes[NodeType.WORKER][0])
        self.assertTrue(manager.all_critical_node_completed())

        for worker in job_nodes[NodeType.WORKER].values():
            worker.reported_status = (NodeEventType.NODE_CHECK_FAILED, 0)
        self.job_context.update_job_nodes(job_nodes)
        self.assertTrue(
            manager._worker_manager.is_all_workers_node_check_failed()
        )

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

        master.perf_monitor.add_running_worker(NodeType.WORKER, 0)
        master.perf_monitor.add_running_worker(NodeType.WORKER, 1)
        cluster_context = ClusterContext(master.job_manager)
        master.perf_monitor.set_target_worker_num(2)
        node.exit_reason = NodeExitReason.FATAL_ERROR
        callback.on_node_failed(node, cluster_context)
        self.assertEqual(master.perf_monitor._target_worker_num, 1)
        self.assertEqual(len(master.perf_monitor.running_workers), 1)
        master.perf_monitor.set_target_worker_num(2)
        master.perf_monitor._workers.add(("worker", 0))
        callback.on_node_succeeded(node, cluster_context)
        self.assertEqual(master.perf_monitor._target_worker_num, 1)

    def test_all_running_node_hang(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()

        hang = manager.all_running_node_hanged()
        self.assertFalse(hang)

        job_nodes = self.job_context.job_nodes()
        for _, nodes in job_nodes.items():
            for _, node in nodes.items():
                node.start_hang_time = time.time() - 3600 * 4
                node.status = NodeStatus.RUNNING
                self.job_context.update_job_node(node)
        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.01, 256)
        hang = manager.all_running_node_hanged()
        self.assertTrue(hang)
        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.5, 256)
        hang = manager.all_running_node_hanged()
        self.assertFalse(hang)

        # test when gpu > 0
        for _, nodes in job_nodes.items():
            for _, node in nodes.items():
                node.start_hang_time = 0
                node.status = NodeStatus.RUNNING
                node.hang = False
                node.config_resource.gpu_num = 1
                self.job_context.update_job_node(node)
        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.01, 256)
        hang = manager.all_running_node_hanged()
        self.assertFalse(hang)
        for _, nodes in job_nodes.items():
            for _, node in nodes.items():
                self.assertFalse(node.start_hang_time)

    def test_no_cpu_request_node_hang(self):
        params = MockK8sJobWithoutCPURequestArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()

        hang = manager.all_running_node_hanged()
        self.assertFalse(hang)

        job_nodes = self.job_context.job_nodes()
        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.01, 256)
        hang = manager.all_running_node_hanged()
        self.assertFalse(hang)

        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.5, 256)
        hang = manager.all_running_node_hanged()
        self.assertFalse(hang)
        for _, nodes in job_nodes.items():
            for _, node in nodes.items():
                node.start_hang_time = time.time() - 3600 * 4
                node.status = NodeStatus.RUNNING
                self.job_context.update_job_node(node)
        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.01, 256)
        hang = manager.all_running_node_hanged()
        self.assertTrue(hang)
        manager.update_node_resource_usage(NodeType.WORKER, 0, 0.5, 256)
        hang = manager.all_running_node_hanged()
        self.assertTrue(hang)

    def test_early_stop_part1(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()

        job_nodes = self.job_context.job_nodes()
        for node in job_nodes[NodeType.PS].values():
            node.status = NodeStatus.PENDING
            node.is_recovered_oom = True
            node.create_time = datetime.now()
            self.job_context.update_job_node(node)
        result, reason, msg = manager.should_early_stop()
        self.assertFalse(result)
        self.assertFalse(reason)
        self.assertFalse(msg)

        manager._remove_exited_node = True
        job_nodes = self.job_context.job_nodes()
        job_nodes[NodeType.WORKER][0].status = NodeStatus.FAILED
        job_nodes[NodeType.WORKER][0].id = 100
        job_nodes[NodeType.WORKER][0].rank_index = 100
        job_nodes[NodeType.WORKER][0].group = 101
        job_nodes[NodeType.WORKER][0].group_size = 1
        job_nodes[NodeType.WORKER][0].group_id = "rack-0"
        self.job_context.update_job_node(job_nodes[NodeType.WORKER][0])
        self.job_context.update_job_node_by_group(
            job_nodes[NodeType.WORKER][0]
        )
        manager.clear_exited_nodes()
        job_nodes = self.job_context.job_nodes()
        self.assertTrue(job_nodes[NodeType.WORKER][0].is_released)

        for node in job_nodes[NodeType.PS].values():
            node.status = NodeStatus.PENDING
            node.create_time = datetime.now() + timedelta(days=-1)
            node.is_recovered_oom = True
            self.job_context.update_job_node(node)
        result, reason, msg = manager.should_early_stop()
        self.assertTrue(result)
        self.assertTrue(reason)
        self.assertTrue(msg)

        job_nodes = self.job_context.job_nodes()
        for node in job_nodes[NodeType.PS].values():
            node.status = NodeStatus.RUNNING
            node.create_time = datetime.now() + timedelta(days=-1)
            node.is_recovered_oom = True
            self.job_context.update_job_node(node)
        result, reason, msg = manager.should_early_stop()
        self.assertFalse(result)
        self.assertFalse(reason)
        self.assertFalse(msg)

    def test_early_stop_part2(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()

        # ps normal + worker pending
        manager._ps_manager.find_pending_node_caused_training_hang = (
            mock.MagicMock(return_value=None)
        )
        manager._worker_manager.find_pending_node_caused_training_hang = (
            mock.MagicMock(return_value="worker0")
        )
        result, reason, msg = manager.should_early_stop()
        self.assertTrue(result)
        self.assertEqual(reason, JobExitReason.PENDING_TIMEOUT)
        self.assertTrue(msg)

        # ps normal + worker normal
        manager._ps_manager.find_pending_node_caused_training_hang = (
            mock.MagicMock(return_value=None)
        )
        manager._worker_manager.find_pending_node_caused_training_hang = (
            mock.MagicMock(return_value=None)
        )
        result, reason, msg = manager.should_early_stop()
        self.assertFalse(result)

        # ps pending + worker normal
        manager._ps_manager.find_pending_node_caused_training_hang = (
            mock.MagicMock(return_value="worker0")
        )
        manager._worker_manager.find_pending_node_caused_training_hang = (
            mock.MagicMock(return_value=None)
        )
        result, reason, msg = manager.should_early_stop()
        self.assertTrue(result)

        # ps pending + worker pending
        manager._ps_manager.find_pending_node_caused_training_hang = (
            mock.MagicMock(return_value="worker0")
        )
        manager._worker_manager.find_pending_node_caused_training_hang = (
            mock.MagicMock(return_value="worker0")
        )
        result, reason, msg = manager.should_early_stop()
        self.assertTrue(result)

    def test_early_stop_part3(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
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

    def test_early_stop_part4(self):
        params = MockK8sAllreduceJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()

        manager._worker_manager.is_all_initial_workers_node_check_failed = (
            mock.MagicMock(return_value=True)
        )
        result, reason, msg = manager.should_early_stop()
        self.assertTrue(result)
        self.assertEqual(reason, JobExitReason.NODE_CHECK_FAILED)

        manager._worker_manager.is_all_initial_workers_node_check_failed = (
            mock.MagicMock(return_value=False)
        )
        manager._worker_manager.is_training_hang_by_pending = mock.MagicMock(
            return_value=False
        )
        manager._worker_manager.is_training_hang_by_insufficient_worker = (
            mock.MagicMock(return_value=False)
        )
        manager._worker_manager.get_pending_node_groups = mock.MagicMock(
            return_value=[]
        )
        manager.handle_node_group_pending()

    def test_when_node_not_init(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        job_context = get_job_context()
        job_nodes = job_context.job_nodes()
        self.assertTrue(len(job_nodes) == 0)

        manager.update_node_resource_usage(NodeType.WORKER, 0, 10, 10240, None)

    def test_start_and_stop(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())

        manager.start()
        node = Node(
            NodeType.WORKER,
            100,
            rank_index=0,
            status=NodeStatus.INITIAL,
            node_group=0,
            node_group_size=1,
            node_group_id="rack0",
            relaunchable=True,
            name="test-0",
        )
        self.job_context.update_job_node(node)
        self.job_context.update_job_node_by_group(node)

        active_threads_name = [t.name for t in threading.enumerate()]
        self.assertIn("node_monitor", active_threads_name)
        self.assertIn("node_heartbeat_monitor", active_threads_name)
        manager.stop()

    def test_concurrency_heart_beat_collecting(self):
        params = MockK8sAllreduceJobArgs()
        worker_size = 1000
        params.initilize(worker_size)
        manager = create_job_manager(params, PerfMonitor())
        manager._scaler._check_master_service_avaliable = mock.MagicMock(
            return_value=True
        )
        manager.start()

        job_nodes = self.job_context.job_nodes()
        self.assertEqual(len(job_nodes[NodeType.WORKER]), worker_size)
        for i, node in job_nodes[NodeType.WORKER].items():
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
        job_nodes = self.job_context.job_nodes()
        for i, node in job_nodes[NodeType.WORKER].items():
            self.assertEqual(node.id, i)
            self.assertEqual(node.heartbeat_time, i)

        manager.stop()
        self.job_context.clear_job_nodes()

        # test when job manager not init
        try:
            manager.collect_node_heart_beat("worker", 1, 111)
        except Exception:
            self.fail()

    def test_get_pending_timeout(self):
        _dlrover_context.seconds_to_wait_pending_pod = 700
        self.assertEqual(get_pending_timeout(), 700)
        _dlrover_context.seconds_to_wait_pending_pod = 0
        self.assertEqual(get_pending_timeout(), 600)
        # reset
        _dlrover_context.seconds_to_wait_pending_pod = 900

    @patch.object(DistributedJobManager, "_report_event")
    def test_process_diagnosis_action(self, mock_method):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())

        manager.process_diagnosis_action(None)
        self.assertEqual(mock_method.call_count, 0)

        manager.process_diagnosis_action(NoAction())
        self.assertEqual(mock_method.call_count, 0)

        manager.process_diagnosis_action(EventAction())
        self.assertEqual(mock_method.call_count, 1)

    def test_process_event_safely(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())

        manager._process_event = mock.MagicMock(side_effect=RuntimeError)
        try:
            manager._process_event_safely(None)
        except Exception:
            self.fail()

    @patch(
        "dlrover.python.master.node.dist_job_manager.DistributedJobManager."
        "_process_event_safely"
    )
    def test_process_node_action(self, mock_process_event):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())

        # no target node
        action = NodeAction(node_id=123, node_type="worker")
        manager._process_node_action(action)

        # with target node
        get_job_context()._job_nodes = {"worker": {0: Node("worker", 0)}}
        action = NodeAction(node_id=0, node_type="worker")
        manager._process_node_action(action)
        mock_process_event.assert_called_once()

    @patch(
        "dlrover.python.master.node.dist_job_manager.DistributedJobManager."
        "_process_event_safely"
    )
    def test_master_restart_with_node_relaunched(self, mock_process_event):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())

        # node0 -> node2 and node1 -> node3 before master restart
        node0 = Node(
            NodeType.WORKER, 0, rank_index=0, status=NodeStatus.INITIAL
        )
        node1 = Node(
            NodeType.WORKER, 1, rank_index=1, status=NodeStatus.INITIAL
        )
        node2 = Node(
            NodeType.WORKER, 2, rank_index=0, status=NodeStatus.RUNNING
        )
        node3 = Node(
            NodeType.WORKER, 3, rank_index=1, status=NodeStatus.RUNNING
        )

        list_nodes = [node2, node3]
        exist_nodes: Dict[str, Dict[int, Node]] = {
            NodeType.WORKER: {0: node0, 1: node1}
        }
        manager.get_job_nodes = mock.MagicMock(return_value=exist_nodes)
        manager._process_list_nodes(list_nodes)

        # assert node0 and node1 got deleted event
        self.assertEqual(mock_process_event.call_count, 4)

        third_call_args, _ = mock_process_event.call_args_list[2]
        third_event = third_call_args[0]
        self.assertEqual(third_event.event_type, NodeEventType.DELETED)
        self.assertEqual(third_event.node.id, 0)
        self.assertEqual(third_event.node.rank_index, 0)

        fourth_call_args, _ = mock_process_event.call_args_list[3]
        forth_event = fourth_call_args[0]
        self.assertEqual(forth_event.event_type, NodeEventType.DELETED)
        self.assertEqual(forth_event.node.id, 1)
        self.assertEqual(forth_event.node.rank_index, 1)

    def test_restart_allreduce_job(self):
        params = MockK8sAllreduceJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()
        manager._scaler.scale = mock.MagicMock(return_value=None)

        # Mock job context methods
        job_context = get_job_context()
        job_context.inc_job_restart_count = mock.MagicMock(return_value=1)
        job_context.request_stop = mock.MagicMock(return_value=None)

        # Set initial job stage
        job_context.update_job_stage(JobStage.JOB_RUNNING)

        # Set worker nodes to various statuses to test different scenarios
        job_nodes = job_context.job_nodes()
        job_nodes[NodeType.WORKER][0].status = NodeStatus.RUNNING
        job_nodes[NodeType.WORKER][0].rank_index = 0
        job_nodes[NodeType.WORKER][1].status = NodeStatus.PENDING
        job_nodes[NodeType.WORKER][1].rank_index = 1
        job_nodes[NodeType.WORKER][2].status = NodeStatus.FINISHED
        job_nodes[NodeType.WORKER][2].rank_index = 2
        job_nodes[NodeType.WORKER][3].status = NodeStatus.FINISHED
        job_nodes[NodeType.WORKER][3].rank_index = 0

        manager.restart()
        manager._scaler.scale.assert_called_once()
        call_args_list = manager._scaler.scale.call_args[0][0]
        self.assertGreaterEqual(len(call_args_list.launch_nodes), 3)

    def test_restart_non_allreduce_job(self):
        params = MockK8sPSJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._scaler.scale = mock.MagicMock(return_value=None)

        manager.restart()
        manager._scaler.scale.assert_not_called()

    def test_restart_over_limit(self):
        params = MockK8sAllreduceJobArgs()
        params.initilize()
        manager = create_job_manager(params, PerfMonitor())
        manager._init_nodes()
        manager._scaler.scale = mock.MagicMock(return_value=None)

        job_context = get_job_context()
        job_context.inc_job_restart_count = mock.MagicMock(return_value=10)
        job_context.request_stop = mock.MagicMock(return_value=None)

        # Set global max restart count to a low value
        from dlrover.python.common.global_context import Context

        dlrover_context = Context.singleton_instance()
        dlrover_context.job_max_restart_count = 5

        manager.restart()
        job_context.request_stop.assert_called_once()
        manager._scaler.scale.assert_not_called()


class JobContextTest(unittest.TestCase):
    def setUp(self):
        self.job_ctx = get_job_context()

    def tearDown(self):
        self.job_ctx.clear_job_node_groups()

    def test_node_group(self):
        self.assertEqual(self.job_ctx.job_node_groups(), {})
        self.assertEqual(list(self.job_ctx.job_node_groups_keys()), [])
        self.assertEqual(self.job_ctx.job_node_group(1), {})
        self.assertIsNone(self.job_ctx.job_group_node_by_rank(1, 0))

        node = Node(
            NodeType.WORKER,
            100,
            rank_index=0,
            status=NodeStatus.INITIAL,
            node_group=0,
            node_group_size=2,
            node_group_id="rack0",
        )
        self.job_ctx.update_job_node_by_group(node)
        self.assertEqual(list(self.job_ctx.job_node_groups_keys()), [0])
        self.assertEqual(list(self.job_ctx.job_node_group(0).keys()), [0])
        self.assertEqual(self.job_ctx.job_group_node_by_rank(0, 0).id, 100)

        node = Node(
            NodeType.WORKER,
            101,
            rank_index=1,
            status=NodeStatus.INITIAL,
            node_group=0,
            node_group_size=2,
            node_group_id="rack0",
        )
        self.job_ctx.update_job_node_by_group(node)
        self.assertEqual(list(self.job_ctx.job_node_groups_keys()), [0])
        self.assertEqual(list(self.job_ctx.job_node_group(0).keys()), [0, 1])
        self.assertEqual(self.job_ctx.job_group_node_by_rank(0, 1).id, 101)

        node = Node(
            NodeType.WORKER,
            200,
            rank_index=2,
            status=NodeStatus.INITIAL,
            node_group=1,
            node_group_size=2,
            node_group_id="rack1",
        )
        self.job_ctx.update_job_node_by_group(node)
        node = Node(
            NodeType.WORKER,
            201,
            rank_index=3,
            status=NodeStatus.INITIAL,
            node_group=1,
            node_group_size=2,
            node_group_id="rack1",
        )
        self.job_ctx.update_job_node_by_group(node)
        self.assertEqual(list(self.job_ctx.job_node_groups_keys()), [0, 1])
        self.assertEqual(list(self.job_ctx.job_node_group(0).keys()), [0, 1])
        self.assertEqual(list(self.job_ctx.job_node_group(1).keys()), [2, 3])


class LocalJobManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.job_context = get_job_context()

    def tearDown(self):
        self.job_context.clear_job_nodes()
        self.job_context.request_stop()

    def test_local_job_manager(self):
        args = LocalJobArgs("local", "default", "test")
        args.initilize()
        args.node_args[NodeType.WORKER].group_resource.count = 4
        job_manager = LocalJobManager(args)
        job_manager.start()

        job_context = get_job_context()
        job_nodes = job_context.job_nodes()
        self.assertEqual(len(job_nodes[NodeType.WORKER]), 4)
        gpu_stats: list[GPUStats] = [
            GPUStats(
                index=0,
                total_memory_mb=24000,
                used_memory_mb=4000,
                gpu_utilization=55.5,
            )
        ]
        job_manager.update_node_resource_usage(
            NodeType.WORKER, 0, 10, 10240, gpu_stats
        )

        job_nodes = job_context.job_nodes()
        worker = job_nodes[NodeType.WORKER][0]
        self.assertEqual(worker.used_resource.cpu, 10)
        self.assertEqual(worker.used_resource.memory, 10240)
        self.assertEqual(worker.used_resource.gpu_stats, gpu_stats)

        dataloader_config = DataLoaderConfig(1, "test_dataloader", 2, 3, 4)
        optimizer_config = OptimizerConfig(1, "test_optimizer", 2)
        paral_config = ParallelConfig(dataloader_config, optimizer_config)
        job_manager.update_node_paral_config(NodeType.WORKER, 0, paral_config)

        job_nodes = job_context.job_nodes()
        worker = job_nodes[NodeType.WORKER][0]
        self.assertEqual(worker.paral_config, paral_config)
        job_manager.handle_training_failure(NodeType.WORKER, 3)

        job_context.set_pre_check_status(PreCheckStatus.FAIL)
        self.assertEqual(job_context.get_pre_check_status(), "FAIL")

        job_context.update_total_worker_num(123)
        self.assertEqual(job_context.get_total_worker_num(), 123)

        self.assertFalse(job_context.is_stopped())
        job_context.request_stop()
        self.assertTrue(job_context.is_stopped())

        job_manager.restart()
        job_manager.stop()
        self.assertEqual(job_manager._stopped, True)

    def test_suspend_unsuspend_job_context(self):
        job_context = get_job_context()

        # test for regular suspend
        init_job_stage = JobStage.JOB_INIT
        job_context.update_job_stage(init_job_stage)

        job_context.request_suspend()
        self.assertEqual(job_context.is_suspended(), True)

        job_context.request_unsuspend()
        self.assertEqual(job_context.get_job_stage(), init_job_stage)

        # try to suspend stopped job, but not work
        job_context.request_stop()
        job_context.request_unsuspend()
        self.assertEqual(job_context.get_job_stage(), JobStage.JOB_STOPPED)

        job_context.request_suspend()
        self.assertEqual(job_context.get_job_stage(), JobStage.JOB_STOPPED)

    def test_job_restart_job_context(self):
        job_context = get_job_context()
        self.assertEqual(job_context.get_job_restart_count(), 0)
        self.assertEqual(job_context.inc_job_restart_count(), 1)
