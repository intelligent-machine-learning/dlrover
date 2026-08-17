# Copyright 2023 The DLRover Authors. All rights reserved.
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

from kubernetes import client

from dlrover.python.common.node import NodeResource
from dlrover.python.scheduler.kubernetes import (
    USER_AGENT,
    get_main_container,
    k8sClient,
    k8sServiceFactory,
    set_container_resource,
)
from dlrover.python.tests.test_utils import create_pod, mock_k8s_client


class KubernetesTest(unittest.TestCase):
    """
    This is a util for testing convenience.
    (To distunguish with 'test_k8s_util')
    """

    def setUp(self) -> None:
        mock_k8s_client()

    def test_get_main_container(self):
        labels = {"class": "test"}
        pod = create_pod(labels)
        main_container = get_main_container(pod)
        self.assertEqual(main_container.name, "main")

        container = client.V1Container(
            name="side",
            image="test",
            command="echo 1",
            resources=client.V1ResourceRequirements(
                requests={},
                limits={},
            ),
            image_pull_policy="Never",
        )

        pod.spec.containers.append(container)
        main_container = get_main_container(pod)
        self.assertEqual(main_container.name, "main")

    def test_set_container_resource(self):
        container = client.V1Container(
            name="side",
            image="test",
            command="echo 1",
            resources=client.V1ResourceRequirements(
                requests={},
                limits={},
            ),
            image_pull_policy="Never",
        )
        resource = NodeResource(4, 1024)
        set_container_resource(container, resource, resource)
        self.assertEqual(container.resources.requests["cpu"], 4)
        self.assertEqual(container.resources.requests["memory"], "1024Mi")
        self.assertEqual(container.resources.limits["cpu"], 4)
        self.assertEqual(container.resources.limits["memory"], "1024Mi")

        container.resources = None
        set_container_resource(container, resource, resource)
        self.assertEqual(container.resources.requests["cpu"], 4)
        self.assertEqual(container.resources.requests["memory"], "1024Mi")
        self.assertEqual(container.resources.limits["cpu"], 4)
        self.assertEqual(container.resources.limits["memory"], "1024Mi")

    def test_service_exists(self):
        fac = k8sServiceFactory("dlrover", "test")
        fac.get_service = unittest.mock.Mock()
        fac.get_service.return_value = fac._create_service_obj(
            "test-master", 12345, 34567, {}, None
        )
        self.assertTrue(
            fac.create_service("test-master", 12345, 34567, {}, None)
        )
        self.assertTrue(
            fac.create_service(
                "test-master", 12345, 34567, {}, None, patch_if_exists=False
            )
        )

    def test_service_factory(self):
        fac = k8sServiceFactory("dlrover", "test")
        self.assertTrue(
            fac.create_service("test-master", 12345, 34567, {}, None)
        )

        svc = fac._create_service_obj("test-master", 12345, 34567, {}, None)
        self.assertEqual(svc.spec.ports[0].name, "grpc")
        svc.spec.ports[0].name = "http"
        succeed = fac._patch_service("test-master", svc, 5)
        self.assertTrue(succeed)
        self.assertEqual(svc.spec.ports[0].name, "http")

    def test_service_obj_with_extra_labels(self):
        fac = k8sServiceFactory("dlrover", "test")

        default_labels = {
            "app": "dlrover",
            "elasticjob.dlrover/name": "test",
        }

        # Without new_labels, only the default labels are applied and the
        # same dict is mirrored into annotations.
        svc = fac._create_service_obj("svc", 12345, 12345, {}, None)
        self.assertEqual(svc.metadata.labels, default_labels)
        self.assertEqual(svc.metadata.annotations, default_labels)

        # new_labels are merged on top; same-named keys override defaults.
        extra = {
            "ignore.extensions.k8s.alipay.com/dnsrr": "true",
            "app": "custom",
        }
        svc = fac._create_service_obj(
            "svc", 12345, 12345, {}, None, new_labels=extra
        )
        self.assertEqual(
            svc.metadata.labels,
            {
                "app": "custom",
                "elasticjob.dlrover/name": "test",
                "ignore.extensions.k8s.alipay.com/dnsrr": "true",
            },
        )
        self.assertEqual(svc.metadata.labels, svc.metadata.annotations)

        # create_service forwards `labels` to _create_service_obj.
        self.assertTrue(
            fac.create_service("svc", 12345, 12345, {}, None, labels=extra)
        )
        # `labels` defaults to None -> no extra keys.
        self.assertTrue(fac.create_service("svc", 12345, 12345, {}, None))

    def test_client(self):
        k8s_client = k8sClient.singleton_instance("default")
        succeed = k8s_client.cordon_node("test-node-0")
        self.assertFalse(succeed)
        self.assertEqual(k8s_client.client.api_client.user_agent, USER_AGENT)
        self.assertEqual(k8s_client.api_client.user_agent, USER_AGENT)
