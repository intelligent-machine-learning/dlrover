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
    get_main_container,
    set_container_resource,
)
from dlrover.python.tests.test_utils import create_pod


class KubernetesTest(unittest.TestCase):
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
