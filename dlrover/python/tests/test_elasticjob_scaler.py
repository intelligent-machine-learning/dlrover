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

from dlrover.python.common.constants import NodeType
from dlrover.python.common.node import Node, NodeGroupResource, NodeResource
from dlrover.python.master.scaler.base_scaler import ScalePlan
from dlrover.python.master.scaler.elasticjob_scaler import ElasticJobScaler
from dlrover.python.tests.test_utils import mock_k8s_client


class ElasticJobScalerTest(unittest.TestCase):
    def test_generate_scaler_crd_by_plan(self):
        mock_k8s_client()
        plan = ScalePlan()
        node_resource = NodeResource(10, 4096)
        plan.launch_nodes.append(
            Node(
                NodeType.WORKER,
                0,
                NodeResource(10, 4096, priority="low"),
                rank_index=0,
                name="test-worker-0",
                service_addr="test-worker-0:2222",
            )
        )
        plan.remove_nodes.append(
            Node(
                NodeType.WORKER,
                1,
                NodeResource(10, 4096, priority="low"),
                rank_index=1,
                name="test-worker-1",
                service_addr="test-worker-1:2222",
            )
        )
        plan.ps_addrs = ["test-ps-0:2222", "test-ps-1:2222"]
        group_resource = NodeGroupResource(1, node_resource)
        plan.node_group_resources["worker"] = group_resource

        scaler = ElasticJobScaler("test", "dlrover")
        scaler_crd = scaler._generate_scale_plan_crd(plan)

        expected_dict = {
            "ownerJob": "test",
            "replicaResourceSpecs": {
                "worker": {
                    "replicas": 1,
                    "resource": {"cpu": "10", "memory": "4096Mi"},
                }
            },
            "createPods": [
                {
                    "name": "test-worker-0",
                    "type": "worker",
                    "id": 0,
                    "rankIndex": 0,
                    "service": "test-worker-0:2222",
                    "resource": {"cpu": "10", "memory": "4096Mi"},
                }
            ],
            "removePods": [
                {
                    "name": "test-worker-1",
                    "type": "worker",
                    "id": 1,
                    "rankIndex": 1,
                    "service": "test-worker-1:2222",
                    "resource": {"cpu": "10", "memory": "4096Mi"},
                }
            ],
            "psHosts": ["test-ps-0:2222", "test-ps-1:2222"],
        }
        print(scaler_crd.spec.to_dict())
        self.assertDictEqual(scaler_crd.spec.to_dict(), expected_dict)
