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

from dlrover.python.common.constants import DistributionStrategy, NodeType
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.scaler.base_scaler import ScalePlan
from dlrover.python.master.scaler.pod_scaler import PodScaler, new_tf_config
from dlrover.python.tests.test_utils import mock_k8s_client


def new_service_fn(node_type, node_id):
    return str(node_type) + "_" + str(node_id)


class PodScalerTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()

    def test_init_pod_template(self):
        scaler = PodScaler("elasticjob-sample", "default")
        self.assertEqual(
            scaler._distribution_strategy,
            DistributionStrategy.PS,
        )
        worker_pod = scaler._replica_template[NodeType.WORKER]
        main_container = worker_pod.spec.containers[0]
        self.assertEqual(
            main_container.image, "dlrover/elasticjob:iris_estimator"
        )
        self.assertEqual(worker_pod.spec.restart_policy, "Never")
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

    def test_create_pod(self):
        scaler = PodScaler("elasticjob-sample", "default")
        scaler._distribution_strategy = DistributionStrategy.PS
        resource = NodeResource(4, 8192)
        node = Node(NodeType.WORKER, 0, resource, rank_index=0)
        pod_stats = {
            NodeType.WORKER: 3,
            NodeType.CHIEF: 1,
            NodeType.PS: 2,
        }
        ps_addrs = [
            "elasticjob-sample-edljob-ps-0",
            "elasticjob-sample-edljob-ps-1",
        ]
        scaler._config_worker_num = 2
        pod = scaler._create_pod(node, pod_stats, ps_addrs)
        self.assertEqual(
            pod.metadata.name, "elasticjob-sample-edljob-worker-0"
        )
        main_container = pod.spec.containers[0]
        self.assertEqual(main_container.resources.limits["cpu"], 4)
        self.assertEqual(main_container.resources.limits["memory"], "8192Mi")
        self.assertEqual(main_container.env[-1].name, "TF_CONFIG")
        self.assertTrue(
            """{"type": "worker", "index": 0}"""
            in main_container.env[-1].value
        )
        self.assertEqual(main_container.env[5].name, "WORKER_NUM")
        self.assertEqual(main_container.env[5].value, "2")
        node = Node(NodeType.CHIEF, 0, resource, rank_index=0)
        pod = scaler._create_pod(node, pod_stats, ps_addrs)
        main_container = pod.spec.containers[0]
        self.assertTrue(
            """{"type": "chief", "index": 0}""" in main_container.env[-1].value
        )

        node = Node(NodeType.PS, 0, resource, rank_index=0)
        pod = scaler._create_pod(node, pod_stats, ps_addrs)
        main_container = pod.spec.containers[0]
        self.assertTrue(
            """{"type": "ps", "index": 0}""" in main_container.env[-1].value
        )

        node = Node(NodeType.WORKER, 0, resource, rank_index=0)
        pod = scaler._create_pod(node, pod_stats, ps_addrs)
        main_container = pod.spec.containers[0]
        self.assertEqual(len(pod.spec.volumes), 1)
        self.assertEqual(pod.spec.volumes[0].name, "pvc-nas")
        self.assertEqual(len(main_container.volume_mounts), 1)

    def test_create_service(self):
        scaler = PodScaler("elasticjob-sample", "default")
        service = scaler._create_service_obj(
            name="elasticjob-sample-edljob-worker-0",
            port="2222",
            target_port="2222",
            replica_type=NodeType.WORKER,
            rank_index=0,
        )
        self.assertEqual(service.spec.selector["rank-index"], "0")
        self.assertEqual(service.spec.selector["replica-type"], "worker")

    def test_scale(self):
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
        scaler.scale(scale_plan)
        self.assertEqual(len(scaler._create_node_queue), 3)

        worker_ids = []
        chief_ids = []
        for node in scaler._create_node_queue:
            if node.type == NodeType.WORKER:
                worker_ids.append(node.id)
            elif node.type == NodeType.CHIEF:
                chief_ids.append(node.id)
        self.assertListEqual(chief_ids, [0])
        self.assertListEqual(worker_ids, [3, 4])

        scale_plan.node_group_resources = {
            NodeType.WORKER: NodeGroupResource(3, resource),
            NodeType.CHIEF: NodeGroupResource(1, resource),
            NodeType.PS: NodeGroupResource(2, resource),
        }
        scale_plan.launch_nodes.append(
            Node(NodeType.WORKER, 1, NodeResource(0, 0))
        )
        scaler.scale(scale_plan)
        self.assertFalse(scale_plan.empty())
        self.assertEqual(len(scaler._create_node_queue), 2)

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
