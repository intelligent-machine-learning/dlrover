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

from dlrover.python.common.constants import NodeStatus, NodeType, PlatformType
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.master.node.ps import ParameterServerManager
from dlrover.python.master.resource.job import JobResource
from dlrover.python.scheduler.factory import new_elastic_job
from dlrover.python.tests.test_utils import mock_k8s_client


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

        plan = self._ps_manager.reduce_pending_node_resource()
        self.assertEqual(len(plan.launch_nodes), 2)
        self.assertEqual(plan.launch_nodes[0].config_resource.cpu, 8)
        self.assertEqual(plan.launch_nodes[0].config_resource.memory, 2048)

    def test_scale_up_ps(self):
        self._ps_manager._scale_up_ps(2)
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
        self.assertEqual(len(plan.remove_nodes), 2)
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
        self.assertEqual(len(plan.launch_nodes), 1)
        self.assertEqual(ps_manager._migrated_ps_nodes[0].id, 2)
        self.assertTrue(ps_manager.exist_migrated_ps_nodes())

        ps_manager._pre_drop_migrated_ps(list(ps_manager._nodes.values()))
        self.assertEqual(len(ps_manager._pre_dropped_ps), 0)
        for node in ps_manager._nodes.values():
            node.status = NodeStatus.RUNNING
        ps_manager._pre_drop_migrated_ps(list(ps_manager._nodes.values()))
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
        ps_manager = ParameterServerManager(
            job_nodes[NodeType.PS],
            self._job_resource,
            3,
            self._elastic_job.get_node_service_addr,
            self._elastic_job.get_node_name,
        )
        for node in ps_manager._nodes.values():
            node.status = NodeStatus.RUNNING
        ps_failure = ps_manager.has_ps_failure()
        self.assertFalse(ps_failure)
        latest_ps_index = len(ps_manager._nodes) - 1
        ps = ps_manager._nodes[latest_ps_index]
        ps_manager._ps_cluster_changed = True
        ps.status = NodeStatus.INITIAL
        ps.init_time -= 600
        ps_failure = ps_manager.has_ps_failure()
        self.assertTrue(ps_failure)
        cluster = ps_manager.get_next_training_ps_cluster()
        self.assertEqual(len(cluster), 1)
        self.assertEqual(
            cluster[0].service_addr,
            "test-edljob-ps-0.default.svc:2222",
        )
