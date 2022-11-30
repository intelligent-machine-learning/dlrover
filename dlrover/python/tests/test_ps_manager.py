# Copyright 2022 The EasyDL Authors. All rights reserved.
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

from dlrover.python.common.constants import EngineType, NodeStatus, NodeType
from dlrover.python.common.node import NodeResource
from dlrover.python.master.node.ps import ParameterServerManager
from dlrover.python.master.resource.job import JobResourceConfig
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.tests.test_utils import mock_k8s_client


class PSManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()
        self._job_resource = JobResourceConfig()
        self._job_resource.add_node_group_resource(
            NodeType.PS, 2, "cpu=16,memory=2048Mi", ""
        )
        self._elastic_job = new_elastic_job(
            EngineType.KUBERNETES, "test", "default"
        )
        self._job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

        self._ps_manager = ParameterServerManager(
            self._job_nodes[NodeType.PS],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )

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
        for _, node in self._ps_manager._nodes.items():
            node.status = NodeStatus.PENDING
            node.create_time = datetime.now() + timedelta(days=-1)

        plan = self._ps_manager.cut_pending_node_cpu()
        self.assertEqual(len(plan.node_resources), 2)
        self.assertEqual(plan.node_resources["test-edljob-ps-0"].cpu, 8)
        self.assertEqual(plan.node_resources["test-edljob-ps-0"].memory, 2048)

    def test_scale_up_ps(self):
        plan = self._ps_manager.scale_up_ps(2)
        self.assertEqual(plan.node_group_resources[NodeType.PS].count, 4)
        self.assertEqual(
            plan.node_group_resources[NodeType.PS].node_resource.cpu, 16
        )
        training_ps = self._ps_manager.get_next_training_ps_cluster()
        self.assertEqual(len(training_ps), 2)
        for node in self._ps_manager._nodes.values():
            node.status = NodeStatus.RUNNING
        training_ps = self._ps_manager.get_next_training_ps_cluster()
        self.assertEqual(len(training_ps), 4)

    def test_scale_down_ps(self):
        job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        ps_manager = ParameterServerManager(
            job_nodes[NodeType.PS],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        for node in ps_manager._nodes.values():
            node.status = NodeStatus.RUNNING
        ps_manager.scale_down_ps(1)
        self.assertEqual(len(ps_manager._pre_dropped_ps), 1)
        self.assertEqual(ps_manager._pre_dropped_ps[0].id, 1)

        plan = ps_manager.process_after_ps_cluster_ready()
        self.assertListEqual(plan.removed_nodes, ["test-edljob-ps-1"])

    def test_delete_running_ps(self):
        job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        ps_manager = ParameterServerManager(
            job_nodes[NodeType.PS],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        for node in ps_manager._nodes.values():
            node.status = NodeStatus.RUNNING

        plan = ps_manager.delete_running_ps()
        self.assertEqual(len(plan.removed_nodes), 2)
        self.assertTrue(job_nodes[NodeType.PS][0].is_released)
        self.assertTrue(job_nodes[NodeType.PS][1].is_released)

    def test_migrate_parameter_servers(self):
        job_nodes = self._job_resource.init_job_node_meta(
            1,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        ps_manager = ParameterServerManager(
            job_nodes[NodeType.PS],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        for node in ps_manager._nodes.values():
            node.status = NodeStatus.RUNNING
        node_name = job_nodes[NodeType.PS][0].name
        nodes = {node_name: NodeResource(20, 2048)}
        plan = ps_manager.migrate_parameter_servers(nodes)
        self.assertEqual(len(plan.node_resources), 1)
        self.assertEqual(ps_manager._migrated_ps_nodes[0].id, 2)
        self.assertTrue(ps_manager.exist_migrated_ps_nodes())

        ps_manager._pre_drop_migrated_ps(list(ps_manager._nodes.values()))
        self.assertEqual(len(ps_manager._pre_dropped_ps), 0)
        for node in ps_manager._nodes.values():
            node.status = NodeStatus.RUNNING
        ps_manager._pre_drop_migrated_ps(list(ps_manager._nodes.values()))
        self.assertEqual(len(ps_manager._pre_dropped_ps), 1)
