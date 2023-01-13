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


class k8sScalerTest(unittest.TestCase):
    def test_generate_scaler_crd_by_plan(self):
        plan = ScalePlan()
        node_resource = NodeResource(10, 4096)
        plan.launch_nodes.append(
            Node(
                NodeType.WORKER,
                0,
                NodeResource(10, 4096, priority="low"),
                rank_index=0,
            )
        )
        group_resource = NodeGroupResource(1, node_resource)
        plan.node_group_resources["worker"] = group_resource

        scaler = ElasticJobScaler("test", "dlrover")
        scaler_crd = scaler._generate_scaler_crd_by_plan(plan)

        expected_dict = {
            "ownerJob": "test",
            "replicaResourceSpec": {
                "worker": {
                    "replicas": 1,
                    "resource": {"cpu": "10", "memory": "4096Mi"},
                }
            },
            "nodeResourceSpec": {"cpu": "10", "memory": "4096Mi"},
        }
        self.assertDictEqual(scaler_crd.spec.to_dict(), expected_dict)
