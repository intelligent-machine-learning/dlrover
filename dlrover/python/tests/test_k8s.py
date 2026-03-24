# Copyright 2026 The DLRover Authors. All rights reserved.
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
from unittest.mock import MagicMock, patch

from kubernetes import client

from dlrover.python.common.constants import (
    NodeType,
    PlatformType,
    k8sAPIExceptionReason,
)
from dlrover.python.scheduler.kubernetes import (
    JOB_SUFFIX,
    NODE_SERVICE_PORTS,
    K8sElasticJob,
    K8sJobArgs,
    convert_cpu_to_decimal,
    convert_memory_to_byte,
    convert_memory_to_mb,
    get_pod_name,
    k8sClient,
    parse_bool,
    retry_k8s_request,
)
from dlrover.python.tests.test_utils import create_pod, mock_k8s_client


class UtilFunctionTest(unittest.TestCase):
    def test_convert_memory_to_mb(self):
        self.assertEqual(convert_memory_to_mb("4096Mi"), 4096)
        self.assertEqual(convert_memory_to_mb("1Gi"), 1024)

    def test_convert_memory_to_byte(self):
        self.assertEqual(convert_memory_to_byte("1Gi"), 1024 * 1024 * 1024)
        self.assertEqual(convert_memory_to_byte("512Mi"), 512 * 1024 * 1024)

    def test_convert_cpu_to_decimal(self):
        self.assertEqual(convert_cpu_to_decimal("1"), 1.0)
        self.assertEqual(convert_cpu_to_decimal("500m"), 0.5)
        self.assertEqual(convert_cpu_to_decimal("2500m"), 2.5)

    def test_parse_bool(self):
        for s in ["true", "True", "yes", "Yes", "t", "T", "y", "Y"]:
            self.assertTrue(parse_bool(s))
        for s in ["false", "no", "0", "n", ""]:
            self.assertFalse(parse_bool(s))

    def test_get_pod_name(self):
        name = get_pod_name("test-job", "worker", 0)
        self.assertEqual(name, "test-job" + JOB_SUFFIX + "worker-0")


class RetryK8sRequestTest(unittest.TestCase):
    def test_retry_returns_on_not_found(self):
        class FakeClient:
            @retry_k8s_request
            def get_something(self):
                raise client.rest.ApiException(
                    status=404, reason=k8sAPIExceptionReason.NOT_FOUND
                )

        result = FakeClient().get_something()
        self.assertIsNone(result)

    @patch("dlrover.python.scheduler.kubernetes.time.sleep")
    def test_retry_exhausted(self, mock_sleep):
        class FakeClient:
            @retry_k8s_request
            def get_something(self):
                raise RuntimeError("fail")

        result = FakeClient().get_something(retry=2)
        self.assertIsNone(result)
        self.assertEqual(mock_sleep.call_count, 2)


class K8sClientTest(unittest.TestCase):
    def setUp(self):
        self.k8s_client = mock_k8s_client()

    def test_create_and_delete_pod(self):
        pod = create_pod({"class": "test"})
        self.assertTrue(self.k8s_client.create_pod(pod))
        self.assertTrue(self.k8s_client.delete_pod("test-worker-0"))

    def test_create_pod_failure(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.client.create_namespaced_pod = MagicMock(
            side_effect=client.rest.ApiException(status=409, reason="Conflict")
        )
        pod = create_pod({"class": "test"})
        # Directly call the real create_pod (not mocked)
        result = k8sClient.create_pod(k8s_client, pod)
        self.assertFalse(result)

    def test_delete_pod_not_found(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.client.delete_namespaced_pod = MagicMock(
            side_effect=client.ApiException(
                status=404, reason=k8sAPIExceptionReason.NOT_FOUND
            )
        )
        result = k8sClient.delete_pod(k8s_client, "nonexist")
        self.assertTrue(result)

    def test_delete_pod_other_error(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.client.delete_namespaced_pod = MagicMock(
            side_effect=client.ApiException(
                status=500, reason="Internal Server Error"
            )
        )
        result = k8sClient.delete_pod(k8s_client, "test-pod")
        self.assertFalse(result)

    def test_create_service_failure(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.client.create_namespaced_service = MagicMock(
            side_effect=client.rest.ApiException(status=409, reason="Conflict")
        )
        svc = client.V1Service(
            metadata=client.V1ObjectMeta(name="test-svc"),
            spec=client.V1ServiceSpec(ports=[]),
        )
        result = k8sClient.create_service(k8s_client, svc)
        self.assertFalse(result)

    def test_patch_service_failure(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.client.patch_namespaced_service = MagicMock(
            side_effect=client.rest.ApiException(
                status=404, reason="Not Found"
            )
        )
        svc = client.V1Service(
            metadata=client.V1ObjectMeta(name="test-svc"),
            spec=client.V1ServiceSpec(ports=[]),
        )
        result = k8sClient.patch_service(k8s_client, "test-svc", svc)
        self.assertFalse(result)

    def test_create_pvc_success(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.client.create_namespaced_persistent_volume_claim = (
            MagicMock(return_value=True)
        )
        pvc = client.V1PersistentVolumeClaim(
            metadata=client.V1ObjectMeta(name="test-pvc")
        )
        result = k8sClient.create_pvc(k8s_client, pvc)
        self.assertTrue(result)

    def test_create_pvc_failure(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.client.create_namespaced_persistent_volume_claim = (
            MagicMock(
                side_effect=client.rest.ApiException(
                    status=409, reason="Conflict"
                )
            )
        )
        pvc = client.V1PersistentVolumeClaim(
            metadata=client.V1ObjectMeta(name="test-pvc")
        )
        result = k8sClient.create_pvc(k8s_client, pvc)
        self.assertFalse(result)

    def test_create_owner_reference(self):
        ref = k8sClient.create_owner_reference(
            "v1alpha1", "ElasticJob", "test-job", "uid-123"
        )
        self.assertEqual(ref.api_version, "v1alpha1")
        self.assertEqual(ref.kind, "ElasticJob")
        self.assertEqual(ref.name, "test-job")
        self.assertEqual(ref.uid, "uid-123")
        self.assertTrue(ref.block_owner_deletion)

    def test_delete_custom_resource(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.api_instance = MagicMock()
        k8s_client.api_instance.delete_namespaced_custom_object = MagicMock()
        k8sClient.delete_custom_resource(
            k8s_client,
            "elastic.iml.github.io",
            "v1alpha1",
            "elasticjobs",
            "test",
        )
        k8s_client.api_instance.delete_namespaced_custom_object.assert_called_once()

    def test_get_pod_annotation(self):
        k8s_client = k8sClient.singleton_instance("default")
        pod = client.V1Pod(
            metadata=client.V1ObjectMeta(
                name="test-pod",
                annotations={"key1": "value1", "key2": "value2"},
            )
        )
        k8s_client.client.read_namespaced_pod = MagicMock(return_value=pod)
        self.assertEqual(
            k8sClient.get_pod_annotation(k8s_client, "test-pod", "key1"),
            "value1",
        )
        self.assertEqual(
            k8sClient.get_pod_annotation(k8s_client, "test-pod", "missing"),
            "",
        )

        # Test with no annotations
        pod.metadata.annotations = None
        self.assertEqual(
            k8sClient.get_pod_annotation(k8s_client, "test-pod", "key1"),
            "",
        )

    def test_delete_custom_resource_not_found(self):
        k8s_client = k8sClient.singleton_instance("default")
        k8s_client.api_instance = MagicMock()
        k8s_client.api_instance.delete_namespaced_custom_object = MagicMock(
            side_effect=client.rest.ApiException(
                status=404, reason=k8sAPIExceptionReason.NOT_FOUND
            )
        )
        # Should not raise
        k8sClient.delete_custom_resource(
            k8s_client,
            "elastic.iml.github.io",
            "v1alpha1",
            "elasticjobs",
            "test",
        )


class K8sElasticJobTest(unittest.TestCase):
    def setUp(self):
        mock_k8s_client()

    def test_get_node_name(self):
        job = K8sElasticJob("test-job", "default")
        name = job.get_node_name(NodeType.WORKER, 0)
        self.assertEqual(name, "test-job" + JOB_SUFFIX + "worker-0")

    def test_get_node_service_addr(self):
        job = K8sElasticJob("test-job", "default")
        addr = job.get_node_service_addr(NodeType.WORKER, 0)
        expected_name = "test-job" + JOB_SUFFIX + "worker-0"
        expected = "%s.%s.svc:%d" % (
            expected_name,
            "default",
            NODE_SERVICE_PORTS[NodeType.WORKER],
        )
        self.assertEqual(addr, expected)

        addr_ps = job.get_node_service_addr(NodeType.PS, 1)
        self.assertIn(":2222", addr_ps)


class K8sJobArgsTest(unittest.TestCase):
    def setUp(self):
        mock_k8s_client()

    def test_initilize(self):
        job_args = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
        job_args.initilize()
        self.assertEqual(job_args.job_uuid, "111-222")
        self.assertEqual(
            job_args.distribution_strategy, "ParameterServerStrategy"
        )
        self.assertTrue(job_args.relaunch_always)
        self.assertIn(NodeType.WORKER, job_args.node_args)
        self.assertIn(NodeType.PS, job_args.node_args)
        self.assertIn(NodeType.CHIEF, job_args.node_args)

        ps_args = job_args.node_args[NodeType.PS]
        self.assertEqual(ps_args.group_resource.count, 3)
        self.assertEqual(ps_args.group_resource.node_resource.cpu, 1.0)
        self.assertEqual(ps_args.group_resource.node_resource.memory, 4096)
        self.assertEqual(ps_args.group_resource.node_resource.priority, "high")
        self.assertEqual(ps_args.restart_count, 3)

    def test_get_job_uuid(self):
        job_args = K8sJobArgs(PlatformType.KUBERNETES, "default", "test")
        uuid = job_args._get_job_uuid({"metadata": {"uid": "abc-123"}})
        self.assertEqual(uuid, "abc-123")

        uuid = job_args._get_job_uuid({"metadata": {}})
        self.assertEqual(uuid, "")

        uuid = job_args._get_job_uuid(None)
        self.assertEqual(uuid, "")


if __name__ == "__main__":
    unittest.main()
