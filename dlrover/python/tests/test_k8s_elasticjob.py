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
from dlrover.python.common.node import NodeResource
from dlrover.python.scheduler.kubernetes import K8sElasticJob
from dlrover.python.tests.test_utils import mock_k8s_client


class K8sElasticJobTest(unittest.TestCase):
    def setUp(self) -> None:
        mock_k8s_client()

    def test_init_pod_template(self):
        job = K8sElasticJob("elasticjob-sample", "default")
        self.assertEqual(job._distribution_strategy, "parameter_server")
        worker_template = job._replica_template[NodeType.WORKER]
        self.assertEqual(
            worker_template.image, "dlrover/elasticjob:iris_estimator"
        )
        self.assertEqual(worker_template.restart_policy, "Never")
        self.assertListEqual(
            worker_template.command,
            [
                "python",
                "-m",
                "model_zoo.iris.dnn_estimator",
                "--batch_size=32",
                "--training_steps=1000",
            ],
        )
        worker0_name = job.get_node_name(NodeType.WORKER, 0)
        self.assertEqual(worker0_name, "elasticjob-sample-edljob-worker-0")

    def test_create_pod(self):
        job = K8sElasticJob("elasticjob-sample", "default")
        worker_resource = NodeResource(4, 8192)
        pod = job.create_typed_pod(NodeType.WORKER, 0, worker_resource)
        self.assertEqual(
            pod.metadata.name, "elasticjob-sample-edljob-worker-0"
        )
        main_container = pod.spec.containers[0]
        self.assertEqual(main_container.resources.limits["cpu"], 4)
        self.assertEqual(main_container.resources.limits["memory"], "8192Mi")

    def test_create_service(self):
        job = K8sElasticJob("elasticjob-sample", "default")
        service = job.create_service(
            NodeType.WORKER, 0, "elasticjob-sample-edljob-worker-0"
        )
        self.assertEqual(service.spec.selector["elastic-replica-index"], "0")
        self.assertEqual(
            service.spec.selector["elastic-replica-type"], "worker"
        )
