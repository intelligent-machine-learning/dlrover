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
from dlrover.python.brain.client import BrainClient


CLIENT_MODULE = "dlrover.python.brain.client"


class TestBrainClient(unittest.TestCase):
    def setUp(self):
        # 1. We mock the protobuf modules so they don't throw ModuleNotFoundErrors
        self.patcher_pb2 = patch(f"{CLIENT_MODULE}.brain_pb2", MagicMock())
        self.mock_pb2 = self.patcher_pb2.start()

        self.patcher_http = patch(f"{CLIENT_MODULE}.http_schemas", MagicMock())
        self.mock_http = self.patcher_http.start()

        # 2. We patch the stubs to prevent actual network connections
        self.patcher_grpc_stub = patch(
            f"{CLIENT_MODULE}.brain_pb2_grpc.BrainStub"
        )
        self.mock_grpc_stub_class = self.patcher_grpc_stub.start()

        self.patcher_http_stub = patch(f"{CLIENT_MODULE}.HttpBrainClient")
        self.mock_http_stub_class = self.patcher_http_stub.start()

    def tearDown(self):
        self.patcher_pb2.stop()
        self.patcher_http.stop()
        self.patcher_grpc_stub.stop()
        self.patcher_http_stub.stop()

    def test_initialization(self):
        """Test that the correct stub is created based on the protocol."""
        # gRPC
        # Assuming _PROTOCOL_GRPC is defined as "grpc" in your module
        client_grpc = BrainClient("localhost:50051", protocol="grpc")
        self.assertTrue(client_grpc.available())
        self.mock_grpc_stub_class.assert_called_once_with("localhost:50051")

        # HTTP
        client_http = BrainClient("http://localhost:8000", protocol="http")
        self.assertTrue(client_http.available())
        self.mock_http_stub_class.assert_called_once_with(
            "http://localhost:8000"
        )

    def test_report_metrics_grpc(self):
        """Test reporting metrics over gRPC routes to persist_metrics."""
        client = BrainClient("localhost:50051", protocol="grpc")
        mock_metrics = MagicMock()

        client.report_metrics(mock_metrics)
        client._brain_stub.persist_metrics.assert_called_once_with(
            mock_metrics
        )

    def test_report_metrics_http(self):
        """Test reporting metrics over HTTP is explicitly bypassed."""
        client = BrainClient("http://localhost", protocol="http")
        mock_metrics = MagicMock()

        result = client.report_metrics(mock_metrics)
        self.assertIsNone(result)
        client._brain_stub.persist_metrics.assert_not_called()

    def test_get_job_metrics(self):
        """Test fetching metrics sends the right proto request."""
        client = BrainClient("localhost", protocol="grpc")
        client.get_job_metrics("job-123")

        # Verify it created a request and assigned the UUID
        self.mock_pb2.JobMetricsRequest.assert_called_once()
        client._brain_stub.get_job_metrics.assert_called_once()

    def test_request_optimization_http_translation(self):
        """Test the complex translation from gRPC protos to HTTP schemas and back."""
        client = BrainClient("http://localhost", protocol="http")

        # 1. Create a fake input gRPC request
        input_req = MagicMock()
        input_req.type = "training"
        input_req.config.brain_processor = "default_processor"
        fake_job = MagicMock(uid="job-1", cluster="prod", namespace="default")
        input_req.jobs = [fake_job]

        # 2. Setup the fake HTTP response we expect from the server
        mock_http_response = MagicMock()
        mock_http_response.response.success = True
        mock_http_response.response.reason = ""

        # Mock the deeply nested node_group_resources
        fake_node_group = MagicMock()
        fake_node_group.count = 4
        fake_node_group.resource.cpu = 8.0
        fake_node_group.resource.memory = 16384
        fake_node_group.resource.gpu = 1.0

        mock_http_response.job_opt_plan.job_resource.node_group_resources.items.return_value = [
            ("worker", fake_node_group)
        ]

        client._brain_stub.optimize.return_value = mock_http_response

        # 3. Call the method
        result = client.request_optimization(input_req)

        # 4. Assertions
        # Verify the HTTP schema was built
        self.mock_http.OptimizeRequest.assert_called_once()
        self.mock_http.JobMeta.assert_called_once_with(
            uuid="job-1", cluster="prod", namespace="default"
        )

        # Verify the stub was called
        client._brain_stub.optimize.assert_called_once()

        # Verify it returned the translated gRPC OptimizeResponse
        self.assertIsNotNone(result)
        self.assertEqual(result.response.success, True)

    @patch(f"{CLIENT_MODULE}.init_job_metrics_message")
    def test_report_training_hyper_params(self, mock_init_metrics):
        """Test that the helper builds the metric correctly before reporting."""
        client = BrainClient("localhost", protocol="grpc")

        # Mock the returned protobuf object from the init function
        mock_job_metrics = MagicMock()
        mock_init_metrics.return_value = mock_job_metrics

        # Setup fake inputs
        job_meta = MagicMock()
        hyper_params = MagicMock(batch_size=32, epoch=10, max_steps=1000)

        # Execute
        client.report_training_hyper_params(job_meta, hyper_params)

        # Assertions
        mock_init_metrics.assert_called_once_with(job_meta)
        self.assertEqual(
            mock_job_metrics.metrics_type,
            self.mock_pb2.MetricsType.Training_Hyper_Params,
        )
        self.assertEqual(mock_job_metrics.training_hyper_params.batch_size, 32)

        # Verify it actually sent it
        client._brain_stub.persist_metrics.assert_called_once_with(
            mock_job_metrics
        )

    def test_get_optimization_plan(self):
        """Test the wrapper that builds an OptimizeRequest for a specific stage."""
        client = BrainClient("localhost", protocol="grpc")

        # Ensure we mock the request_optimization so it doesn't actually run
        client.request_optimization = MagicMock()

        client.get_optimization_plan(
            "job-123", "stage-1", "my_retriever", {"custom_key": "val"}
        )

        # Check that the underlying call was made
        client.request_optimization.assert_called_once()

        # Retrieve the arguments passed to request_optimization
        args, _ = client.request_optimization.call_args
        passed_req = args[0]

        self.assertEqual(passed_req.type, "stage-1")
        self.assertEqual(
            passed_req.config.optimizer_config_retriever, "my_retriever"
        )
        # Assuming jobs is a mocked protobuf repeated field, checking the item
        self.assertEqual(passed_req.jobs[0].uid, "job-123")

    def test_get_config_success(self):
        """Test the get_config method properly extracts the value."""
        client = BrainClient("localhost", protocol="grpc")

        # Setup fake successful response
        fake_response = MagicMock()
        fake_response.response.success = True
        fake_response.config_value = "my_custom_value"
        client._brain_stub.get_config.return_value = fake_response

        result = client.get_config("my_key")

        self.assertEqual(result, "my_custom_value")
        self.mock_pb2.ConfigRequest.assert_called_once()

    def test_get_config_failure(self):
        """Test get_config returns None if the server says success=False."""
        client = BrainClient("localhost", protocol="grpc")

        fake_response = MagicMock()
        fake_response.response.success = False
        client._brain_stub.get_config.return_value = fake_response

        result = client.get_config("bad_key")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
