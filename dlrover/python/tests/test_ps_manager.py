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

import unittest
from datetime import datetime, timedelta

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeStatus,
    NodeType,
    PlatformType,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.master.node.ps import ParameterServerManager
from dlrover.python.master.resource.job import JobResource
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.tests.test_utils import mock_k8s_client

_dlrover_ctx = Context.singleton_instance()


class PSManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        self._job_resource = JobResource()
        self._job_resource.node_group_resources[
            NodeType.PS
        ] = NodeGroupResource(2, NodeResource(16, 2048))
        self._elastic_job = new_elastic_job(
            PlatformType.KUBERNETES, "test", "default"
        )
        self._job_context = get_job_context()
        job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._job_context.update_job_nodes(job_nodes)

        self._ps_manager = ParameterServerManager(
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

    def tearDown(self) -> None:
        self._job_context.clear_job_nodes()

    def test_get_training_ps_cluster(self):
        ps_nodes = self._ps_manager.get_training_ps_cluster()
        self.assertEqual(len(ps_nodes), 2)
        ps_hosts = self._ps_manager.get_ps_addrs()
        self.assertListEqual(
            ps_hosts,
            [
                "test-edljob-ps-0.default.svc:2222",
                "test-edljob-ps-1.default.svc:2222",
            ],
        )

    def test_cut_pending_ps_cpu(self):
        nodes = self._job_context.job_nodes_by_type(NodeType.PS)
        for _, node in nodes.items():
            node.status = NodeStatus.PENDING
            node.create_time = datetime.now() + timedelta(days=-1)
            self._job_context.update_job_node(node)

        plan = self._ps_manager.reduce_pending_node_resource()
        self.assertEqual(len(plan.launch_nodes), 2)
        self.assertEqual(plan.launch_nodes[0].config_resource.cpu, 8)
        self.assertEqual(plan.launch_nodes[0].config_resource.memory, 2048)

    def test_scale_up_ps(self):
        self._ps_manager._scale_up_ps(2)
        training_ps = self._ps_manager.get_next_training_ps_cluster()
        self.assertEqual(len(training_ps), 2)

        nodes = self._job_context.job_nodes_by_type(NodeType.PS)
        for node in nodes.values():
            node.status = NodeStatus.RUNNING
            self._job_context.update_job_node(node)
        training_ps = self._ps_manager.get_next_training_ps_cluster()
        self.assertEqual(len(training_ps), 4)

    def test_scale_down_ps(self):
        job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._job_context.update_job_nodes(job_nodes)
        ps_manager = ParameterServerManager(
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        nodes = self._job_context.job_nodes_by_type(NodeType.PS)
        for node in nodes.values():
            node.status = NodeStatus.RUNNING
            self._job_context.update_job_node(node)
        ps_manager._scale_down_ps(1)
        self.assertEqual(len(ps_manager._pre_dropped_ps), 1)
        self.assertEqual(ps_manager._pre_dropped_ps[0].id, 1)

        plan = ps_manager.process_after_ps_cluster_ready()
        self.assertEqual(plan.remove_nodes[0].name, "test-edljob-ps-1")

    def test_delete_running_ps(self):
        job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._job_context.update_job_nodes(job_nodes)
        ps_manager = ParameterServerManager(
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        nodes = self._job_context.job_nodes_by_type(NodeType.PS)
        for node in nodes.values():
            node.status = NodeStatus.RUNNING
            self._job_context.update_job_node(node)

        plan = ps_manager.delete_running_ps()
        job_nodes = self._job_context.job_nodes()
        self.assertEqual(len(plan.remove_nodes), 2)
        self.assertTrue(job_nodes[NodeType.PS][0].is_released)
        self.assertTrue(job_nodes[NodeType.PS][1].is_released)

    def test_migrate_parameter_servers(self):
        job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._job_context.update_job_nodes(job_nodes)
        ps_manager = ParameterServerManager(
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

        nodes = self._job_context.get_mutable_ps_nodes()
        for node in nodes.values():
            node.status = NodeStatus.RUNNING
            self._job_context.update_job_node(node)

        job_nodes = self._job_context.job_nodes()
        node_name = job_nodes[NodeType.PS][0].name
        nodes = {node_name: NodeResource(20, 2048)}
        plan = ps_manager.migrate_parameter_servers(nodes)
        self.assertEqual(len(plan.launch_nodes), 1)
        self.assertEqual(ps_manager._migrated_ps_nodes[0].id, 2)
        self.assertTrue(ps_manager.exist_migrated_ps_nodes())

        nodes = self._job_context.get_mutable_ps_nodes()
        ps_manager._pre_drop_migrated_ps(list(nodes.values()))
        self.assertEqual(len(ps_manager._pre_dropped_ps), 0)
        for node in nodes.values():
            node.status = NodeStatus.RUNNING
            self._job_context.update_job_node(node)
        nodes = self._job_context.get_mutable_ps_nodes()
        ps_manager._pre_drop_migrated_ps(list(nodes.values()))
        self.assertEqual(len(ps_manager._pre_dropped_ps), 1)

        training_ps = ps_manager.get_next_training_ps_cluster()
        self.assertEqual(len(training_ps), 2)
        self.assertEqual(
            training_ps[0].service_addr, "test-edljob-ps-2.default.svc:2222"
        )

    def test_parameter_server_failure(self):
        job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self._job_context.update_job_nodes(job_nodes)
        ps_manager = ParameterServerManager(
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        nodes = self._job_context.job_nodes_by_type(NodeType.PS)
        for node in nodes.values():
            node.status = NodeStatus.RUNNING
            self._job_context.update_job_node(node)
        ps_failure = ps_manager.has_ps_failure()
        self.assertFalse(ps_failure)
        latest_ps_index = len(nodes) - 1
        ps = nodes[latest_ps_index]
        ps_manager._ps_cluster_changed = True
        ps.status = NodeStatus.INITIAL
        ps.init_time -= 600
        self._job_context.update_job_node(ps)
        ps_failure = ps_manager.has_ps_failure()
        self.assertTrue(ps_failure)
        cluster = ps_manager.get_next_training_ps_cluster()
        self.assertEqual(len(cluster), 1)
        self.assertEqual(
            cluster[0].service_addr,
            "test-edljob-ps-0.default.svc:2222",
        )

    def test_is_training_hang_by_pending_ps(self):
        _dlrover_ctx.pending_fail_strategy = 1
        ps_manager = ParameterServerManager(
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        self.assertFalse(
            ps_manager.is_training_hang_by_pending(
                4, DistributionStrategy.ALLREDUCE
            )
        )
        self.assertFalse(
            ps_manager.is_training_hang_by_pending(4, DistributionStrategy.PS)
        )

        mock_nodes = {}

        # =========================================
        # condition: when node required is updated
        # =========================================

        # mock with 3 running + 1 pending short time
        ps_num = 4
        for index in range(4):
            mock_node = Node(
                NodeType.PS,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-1)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            mock_nodes[index] = mock_node
            self._job_context.update_job_node(mock_node)
        self.assertFalse(
            ps_manager.is_training_hang_by_pending(
                ps_num, DistributionStrategy.ALLREDUCE
            )
        )
        self.assertFalse(
            ps_manager.is_training_hang_by_pending(
                ps_num, DistributionStrategy.PS
            )
        )
        mock_nodes.clear()
        self._job_context.clear_job_nodes()

        # mock with 3 running + 1 pending long time
        for index in range(4):
            mock_node = Node(
                NodeType.PS,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            if index == 0:
                mock_node.status = NodeStatus.PENDING
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            else:
                mock_node.create_time = datetime.now() + timedelta(minutes=-20)
            mock_nodes[index] = mock_node
            self._job_context.update_job_node(mock_node)
        self.assertFalse(
            ps_manager.is_training_hang_by_pending(
                ps_num, DistributionStrategy.ALLREDUCE
            )
        )
        self.assertTrue(
            ps_manager.is_training_hang_by_pending(
                ps_num, DistributionStrategy.PS
            )
        )
        mock_nodes.clear()
        self._job_context.clear_job_nodes()

        # mock with 4 running
        for index in range(4):
            mock_node = Node(
                NodeType.PS,
                index,
                NodeResource(0, 0),
                "test-" + str(index),
                NodeStatus.RUNNING,
            )
            mock_nodes[index] = mock_node
            self._job_context.update_job_node(mock_node)
        self.assertFalse(
            ps_manager.is_training_hang_by_pending(
                ps_num, DistributionStrategy.ALLREDUCE
            )
        )
        self.assertFalse(
            ps_manager.is_training_hang_by_pending(
                ps_num, DistributionStrategy.PS
            )
        )
        mock_nodes.clear()
        self._job_context.clear_job_nodes()
