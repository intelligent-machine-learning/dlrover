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
from collections import deque
from threading import Thread
from unittest import mock
from unittest.mock import MagicMock

from dlrover.python.common.constants import (
    DistributionStrategy,
    NodeStatus,
    NodeType,
    SchedulingLabel,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.scaler.base_scaler import ScalePlan
from dlrover.python.master.scaler.pod_scaler import PodScaler, new_tf_config
from dlrover.python.tests.test_utils import mock_k8s_client

_dlrover_ctx = Context.singleton_instance()


def new_service_fn(node_type, node_id):
    return str(node_type) + "_" + str(node_id)


class PodScalerTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["POD_IP"] = "127.0.0.1"
        mock_k8s_client()

    def tearDown(self) -> None:
        os.environ.clear()

    def test_init_pod_template(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._check_master_service_avaliable = mock.MagicMock(
            return_value=True
        )
        scaler.start()
        self.assertEqual(
            scaler._distribution_strategy,
            DistributionStrategy.PS,
        )
        worker_pod = scaler._replica_template[NodeType.WORKER]
        main_container = worker_pod.spec.containers[0]
        self.assertEqual(
            main_container.image, "dlrover/elasticjob:iris_estimator"
        )
        self.assertEqual(worker_pod.spec.restart_policy, "Always")
        self.assertListEqual(
            main_container.command,
            [
                "python",
                "-m",
                "model_zoo.iris.dnn_estimator",
                "--batch_size=32",
                "--training_steps=1000",
            ],
        )

    def test_check_master_service_avaliable(self):
        scaler = PodScaler("elasticjob-sample", "default")
        _dlrover_ctx.config_master_port()
        port = _dlrover_ctx.master_port
        if 22222 == port:
            wrong_port = 11111
        else:
            wrong_port = 22222
        passed = scaler._check_master_service_avaliable(
            "elasticjob-test-master", wrong_port, 1
        )
        self.assertFalse(passed)

        passed = scaler._check_master_service_avaliable(
            "localhost", wrong_port, 1
        )
        self.assertFalse(passed)

        passed = scaler._check_master_service_avaliable("localhost", port, 1)
        self.assertFalse(passed)

    def test_periodic_create_pod(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._check_master_service_avaliable = unittest.mock.MagicMock(
            return_value=True
        )
        scaler._create_pod = unittest.mock.MagicMock(return_value=True)
        scaler._create_service_for_pod = unittest.mock.MagicMock(
            return_value=True
        )

        scaler._create_node_queue = deque()
        test_num = 10
        for i in range(test_num):
            node = Node(
                NodeType.WORKER, i, NodeResource(4, 8192), rank_index=i
            )
            scaler._create_node_queue.append(node)

        scaler.start()
        time.sleep(3)
        scaler.stop()
        self.assertEqual(scaler._create_pod.call_count, test_num)
        self.assertEqual(scaler._create_service_for_pod.call_count, test_num)

    def test_create_pod(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._check_master_service_avaliable = mock.MagicMock(
            return_value=True
        )
        _dlrover_ctx.config_master_port()

        scaler.start()
        scaler._init_pod_config_by_job()
        scaler._distribution_strategy = DistributionStrategy.PS
        resource = NodeResource(4, 8192)
        node = Node(NodeType.WORKER, 0, resource, rank_index=0)
        node.group = 1024
        node.group_size = 1
        node.group_id = "rack-0"

        # mock field
        scaler._pod_stats = {
            NodeType.WORKER: 3,
            NodeType.CHIEF: 1,
            NodeType.PS: 2,
        }
        scaler._ps_addrs = [
            "elasticjob-sample-edljob-ps-0",
            "elasticjob-sample-edljob-ps-1",
        ]
        scaler._config_worker_num = 2
        pod = scaler._create_pod(node)
        self.assertEqual(
            pod.metadata.name, "elasticjob-sample-edljob-worker-0"
        )
        self.assertEqual(
            pod.metadata.labels[SchedulingLabel.NODE_GROUP], str(node.group)
        )

        main_container = pod.spec.containers[0]
        self.assertEqual(main_container.resources.limits["cpu"], 4)
        self.assertEqual(main_container.resources.limits["memory"], "8192Mi")
        self.assertEqual(main_container.env[-1].name, "TF_CONFIG")
        self.assertTrue(
            """{"type": "worker", "index": 0}"""
            in main_container.env[-1].value
        )
        env_worker_num = 0
        env_job_name = ""
        env_job_uid = ""
        host_ports = ""
        for env in main_container.env:
            if env.name == "WORKER_NUM":
                env_worker_num = int(env.value)
            elif env.name == "ELASTIC_JOB_NAME":
                env_job_name = env.value
            elif env.name == "JOB_UID":
                env_job_uid = env.value
            elif env.name == "HOST_PORTS":
                host_ports = env.value

        self.assertEqual(env_worker_num, 2)
        self.assertEqual(env_job_name, "elasticjob-sample")
        self.assertEqual(env_job_uid, "111-222")
        self.assertEqual(host_ports, "1,2,3,4,5")

        node = Node(NodeType.CHIEF, 0, resource, rank_index=0)
        pod = scaler._create_pod(node)
        main_container = pod.spec.containers[0]
        self.assertTrue(
            """{"type": "chief", "index": 0}""" in main_container.env[-1].value
        )
        node = Node(NodeType.PS, 0, resource, rank_index=0)
        pod = scaler._create_pod(node)
        main_container = pod.spec.containers[0]
        self.assertTrue(
            """{"type": "ps", "index": 0}""" in main_container.env[-1].value
        )

        node = Node(NodeType.WORKER, 0, resource, rank_index=0)
        pod = scaler._create_pod(node)
        main_container = pod.spec.containers[0]
        self.assertEqual(len(pod.spec.volumes), 1)
        self.assertEqual(pod.spec.volumes[0].name, "pvc-nas")
        self.assertEqual(len(main_container.volume_mounts), 1)

        scaler._distribution_strategy = DistributionStrategy.ALLREDUCE
        node = Node(NodeType.WORKER, 0, resource, rank_index=0)
        scaler._ps_addrs = []
        pod = scaler._create_pod(node)
        main_container = pod.spec.containers[0]
        world_size = -1
        rank = -1
        for env in main_container.env:
            if env.name == "WORLD_SIZE":
                world_size = int(env.value)
            elif env.name == "RANK":
                rank = int(env.value)
        self.assertEqual(world_size, 2)
        self.assertEqual(rank, 0)

    def test_create_pod_from_queue(self):
        scaler = PodScaler("elasticjob-sample", "default")
        worker0 = Node(
            NodeType.WORKER,
            0,
            NodeResource(1, 1024, gpu_num=1),
            name="test-worker-0",
        )
        scaler._pod_stats = {}
        scaler._pod_stats[NodeType.WORKER] = 3
        scaler._create_node_queue.append(worker0)
        scaler._create_service_for_pod = mock.MagicMock(  # type: ignore
            return_value=True
        )
        scaler._create_pod_from_queue(scaler._create_node_queue.popleft())
        self.assertEqual(len(scaler._create_node_queue), 0)

        scaler._create_service_for_pod = mock.MagicMock(  # type: ignore
            return_value=False
        )
        worker1 = Node(
            NodeType.WORKER,
            1,
            NodeResource(1, 1024),
            name="test-worker-0",
        )
        scaler._create_node_queue.extend([worker0, worker1])

        scaler._create_pod = mock.MagicMock(side_effect=Exception())
        scaler._create_pod_from_queue(scaler._create_node_queue.popleft())

    def test_scale(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._started = True
        scaler._distribution_strategy = DistributionStrategy.PS
        resource = NodeResource(4, 8192)

        scaler._elasticjob_exists = mock.MagicMock(return_value=True)
        scaler.scale(ScalePlan())

        scale_plan = ScalePlan()
        self.assertTrue(scale_plan.empty())
        scale_plan.node_group_resources = {
            NodeType.WORKER: NodeGroupResource(5, resource),
            NodeType.CHIEF: NodeGroupResource(1, resource),
            NodeType.PS: NodeGroupResource(2, resource),
        }
        scale_plan.ps_addrs = ["ps-0:22222"]
        scaler.scale(scale_plan)
        self.assertEqual(len(scaler._create_node_queue), 3)
        self.assertListEqual(scaler._ps_addrs, scale_plan.ps_addrs)

        worker_ids = []
        chief_ids = []
        for node in scaler._create_node_queue:
            if node.type == NodeType.WORKER:
                worker_ids.append(node.id)
            elif node.type == NodeType.CHIEF:
                chief_ids.append(node.id)
        self.assertListEqual(chief_ids, [0])
        self.assertListEqual(worker_ids, [3, 4])
        scaler._create_node_queue.clear()

        scale_plan.node_group_resources = {
            NodeType.WORKER: NodeGroupResource(3, resource),
            NodeType.CHIEF: NodeGroupResource(1, resource),
            NodeType.PS: NodeGroupResource(2, resource),
        }
        scale_plan.launch_nodes.append(
            Node(NodeType.WORKER, 1, NodeResource(0, 0))
        )
        scale_plan.remove_nodes.append(
            Node(NodeType.WORKER, 3, NodeResource(0, 0))
        )
        plan_json = scale_plan.to_json()
        self.assertTrue("paral_config" not in plan_json)
        scaler.scale(scale_plan)
        self.assertFalse(scale_plan.empty())
        self.assertEqual(len(scaler._create_node_queue), 2)

        # test wait async execution
        scaler._create_node_queue.clear()
        mock_future = MagicMock()
        mock_future.done.return_value = False
        scaler._create_node_futures.append(mock_future)

        thread = Thread(target=scaler.scale, args=(scale_plan,))
        thread.start()
        thread.join(timeout=1)
        self.assertTrue(thread.is_alive())
        scaler._started = False

    def test_scale_thread(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._check_master_service_avaliable = mock.MagicMock(
            return_value=True
        )
        scaler.start()
        scaler._distribution_strategy = DistributionStrategy.PS
        resource = NodeResource(4, 8192)
        scale_plan = ScalePlan()
        scale_plan.launch_nodes.append(Node(NodeType.WORKER, 1, resource))
        scale_plan.ps_addrs = ["ps-0:22222"]
        scaler.scale(scale_plan)
        self.assertEqual(len(scaler._create_node_queue), 1)
        scaler._create_node_queue.clear()
        scale_plan = ScalePlan()
        scale_plan.launch_nodes.append(Node(NodeType.WORKER, 2, resource))
        scale_plan.ps_addrs = ["ps-0:22222"]
        scaler.scale(scale_plan)
        self.assertEqual(len(scaler._create_node_queue), 1)
        scaler._create_node_queue.clear()

    def test_new_tf_config(self):
        pod_stats = {NodeType.WORKER: 1}

        tf_config = new_tf_config(
            pod_stats, new_service_fn, NodeType.WORKER, 0, []
        )
        self.assertDictEqual(
            tf_config,
            {
                "cluster": {"ps": [], "worker": ["worker_0"]},
                "task": {"type": "worker", "index": 0},
            },
        )

    def test_scale_up_pods(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._distribution_strategy = DistributionStrategy.PS
        resource = NodeResource(4, 8192)
        scale_plan = ScalePlan()
        self.assertTrue(scale_plan.empty())
        scale_plan.node_group_resources = {
            NodeType.WORKER: NodeGroupResource(5, resource),
            NodeType.CHIEF: NodeGroupResource(1, resource),
            NodeType.PS: NodeGroupResource(2, resource),
        }
        cur_nodes = [Node(NodeType.WORKER, 1, rank_index=0)]
        scaler._scale_up_pods(NodeType.WORKER, scale_plan, cur_nodes, 1)
        self.assertEqual(len(scaler._create_node_queue), 4)
        self.assertEqual(
            scaler._create_node_queue[0].service_addr,
            "elasticjob-sample-edljob-worker-1.default.svc:3333",
        )

    def test_update_job_pods(self):
        scaler = PodScaler("elasticjob-sample", "default")

        test_nodes = [
            Node(NodeType.WORKER, 0, rank_index=0, status=NodeStatus.RUNNING),
            Node(NodeType.WORKER, 1, rank_index=1, status=NodeStatus.RUNNING),
            Node(NodeType.WORKER, 2, rank_index=2, status=NodeStatus.FAILED),
            Node(NodeType.WORKER, 3, rank_index=3, status=NodeStatus.RUNNING),
        ]
        job_pods = {NodeType.WORKER: test_nodes}
        scaler._update_job_pods(job_pods)
        self.assertEqual(scaler._safe_get_pod_status(NodeType.WORKER, 0), 4)
        self.assertEqual(
            scaler._safe_get_alive_pod_status(NodeType.WORKER, 0), 3
        )

    def test_multi_scales_with_merge(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._check_master_service_avaliable = mock.MagicMock(
            return_value=True
        )
        scaler._elasticjob_exists = mock.MagicMock(return_value=True)
        scaler._job_context.is_suspended = mock.MagicMock(return_value=False)
        scaler._remove_nodes = mock.MagicMock()
        scaler.start()

        resource = NodeResource(4, 8192)

        first_last_time = 1
        second_last_time = 1
        third_last_time = 1
        try:
            scale_plan1 = ScalePlan()
            scale_plan1.launch_nodes.append(
                Node(NodeType.WORKER, 0, resource, relaunch_count=1)
            )

            start_time = time.time()
            scaler.scale(scale_plan1, with_merge=True)
            first_last_time = time.time() - start_time
            self.assertFalse(scaler._pending_plan)

            scale_plan2 = ScalePlan()
            scale_plan2.launch_nodes.append(
                Node(NodeType.WORKER, 1, resource, relaunch_count=1)
            )

            start_time = time.time()
            scaler.scale(scale_plan2, with_merge=True)
            second_last_time = time.time() - start_time
            self.assertTrue(scaler._pending_plan)
            self.assertTrue(len(scaler._pending_plan.launch_nodes), 1)

            scale_plan3 = ScalePlan()
            scale_plan3.launch_nodes.append(
                Node(NodeType.WORKER, 2, resource, relaunch_count=1)
            )

            start_time = time.time()
            scaler.scale(scale_plan3, with_merge=True)
            third_last_time = time.time() - start_time
            self.assertTrue(scaler._pending_plan)
            self.assertTrue(len(scaler._pending_plan.launch_nodes), 2)
        finally:
            self.assertTrue(first_last_time < 1)
            self.assertTrue(second_last_time < 1)
            self.assertTrue(third_last_time < 1)
            scaler.stop()

    def test_execute_scale_with_pending(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._check_master_service_avaliable = mock.MagicMock(
            return_value=True
        )
        scaler._elasticjob_exists = mock.MagicMock(return_value=True)
        scaler._job_context.is_suspended = mock.MagicMock(return_value=False)
        scaler._remove_nodes = mock.MagicMock()
        scaler._scale = mock.MagicMock()
        scaler.start()

        try:
            pending_plan = ScalePlan()
            pending_plan.launch_nodes.append(
                Node(NodeType.WORKER, 1, NodeResource(1, 1024))
            )
            scaler._pending_plan = pending_plan
            scaler._execute_scale_with_pending(ScalePlan())

            self.assertIsNone(scaler._pending_plan)
            self.assertEqual(scaler._scale.call_count, 2)  # initial + pending
        finally:
            scaler.stop()
