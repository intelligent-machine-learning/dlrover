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

import dlrover.python.util.k8s_util as ku
from dlrover.python.common.constants import ElasticJobLabel, NodeType


class KubernetesUtilTest(unittest.TestCase):
    def test_kubernetes_label_selector_transformation(self):
        master_labels = {
            ElasticJobLabel.JOB_KEY: "ut-test",
            ElasticJobLabel.REPLICA_TYPE_KEY: NodeType.DLROVER_MASTER,
        }
        selector_result = ku.gen_k8s_label_selector_from_dict(master_labels)
        self.assertTrue("elasticjob.dlrover/name=ut-test" in selector_result)
        self.assertTrue(
            "elasticjob.dlrover/replica-type=dlrover-master" in selector_result
        )

        dict_result = ku.gen_dict_from_k8s_label_selector(selector_result)
        self.assertEqual(master_labels, dict_result)
