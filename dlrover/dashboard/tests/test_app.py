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

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application

# Import the module to test
from dlrover.dashboard.app import (
    BaseHandler,
    create_dashboard_app,
)


class TestBaseHandler(AsyncHTTPTestCase):
    """Test BaseHandler functionality."""

    def get_app(self):
        return Application([("/", BaseHandler)])

    def test_set_default_headers(self):
        """Test CORS headers are properly set."""
        response = self.fetch("/", method="OPTIONS")
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Origin"), "*"
        )
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Headers"),
            "x-requested-with",
        )
        self.assertEqual(
            response.headers.get("Access-Control-Allow-Methods"),
            "POST, GET, OPTIONS",
        )


class TestJobInfoHandler(AsyncHTTPTestCase):
    """Test JobInfoHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_job_info_success(self, mock_get_job_context, mock_json):
        """Test successful retrieval of job information."""
        # Mock job context
        mock_job_ctx = Mock()
        mock_job_ctx.get_job_stage.return_value = "RUNNING"
        mock_job_ctx.get_failed_node_cnt.return_value = 1
        mock_job_ctx.job_nodes_by_type.return_value = []
        mock_get_job_context.return_value = mock_job_ctx
        mock_json.dumps.return_value = {
            "job_stage": "RUNNING",
            "failed_nodes": 1,
            "job_nodes_by_type": [],
        }

        response = self.fetch("/api/job")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("job_stage", data)
        self.assertEqual(data["job_stage"], "RUNNING")
        self.assertIn("failed_nodes", data)
        self.assertEqual(data["failed_nodes"], 1)

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_job_info_with_attributes(
        self, mock_get_job_context, mock_json
    ):
        """Test job info retrieval with various attributes."""
        mock_job_ctx = Mock()
        mock_job_ctx._job_name = "test-job"
        mock_job_ctx._job_type = "tensorflow"
        mock_job_ctx._job_create_time = None
        mock_job_ctx._job_start_time = None
        mock_job_ctx.get_job_stage.return_value = "COMPLETED"
        mock_job_ctx.get_failed_node_cnt.return_value = 0
        mock_job_ctx.job_nodes_by_type.return_value = []
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "job_name": "test-job",
            "job_type": "tensorflow",
            "job_stage": "COMPLETED",
            "failed_nodes": 0,
            "create_time": "2026-01-01T00:00:00",
            "start_time": "2026-01-01T00:00:00",
            "total_nodes": 0,
            "running_nodes": 0,
            "succeeded_nodes": 0,
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/job")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertEqual(data["job_name"], "test-job")
        self.assertEqual(data["job_type"], "tensorflow")
        self.assertEqual(data["job_stage"], "COMPLETED")
        self.assertEqual(data["failed_nodes"], 0)


class TestNodesHandler(AsyncHTTPTestCase):
    """Test NodesHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.NodesHandler._build_consanguinity_chain")
    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_nodes_success(
        self, mock_get_job_context, mock_json, mock_build_consanguinity
    ):
        """Test successful retrieval of node information."""
        # Mock _build_consanguinity_chain to return empty string
        mock_build_consanguinity.return_value = ""

        # Create mock node
        mock_node = Mock()
        mock_node.id = 1
        mock_node.type = "WORKER"
        mock_node.status = "RUNNING"
        mock_node.name = "worker-1"
        mock_node.rank_index = 0
        mock_node.service_addr = "10.0.0.1:8080"
        mock_node.start_time = None
        mock_node.finish_time = None
        mock_node.exit_reason = None
        mock_node.relaunch_count = 0
        mock_node.config_resource = None
        mock_node.used_resource = None
        mock_node.critical = True
        mock_node.max_relaunch_count = 3
        mock_node.unrecoverable_failure_msg = None
        mock_node.host_name = "worker-1-host"
        mock_node.host_ip = "10.0.0.1"

        # Mock job context
        mock_job_ctx = Mock()
        mock_job_ctx.get_mutable_job_nodes.return_value = {1: mock_node}
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "worker": [
                {
                    "id": 1,
                    "type": "worker",
                    "status": "RUNNING",
                    "name": "worker-1",
                    "rank_index": 0,
                    "service_addr": "10.0.0.1:8080",
                    "start_time": None,
                    "finish_time": None,
                    "exit_reason": None,
                    "relaunch_count": 0,
                    "config_resource": None,
                    "used_resource": None,
                    "critical": True,
                    "max_relaunch_count": 3,
                    "unrecoverable_failure_msg": None,
                    "hostname": "worker-1-host",
                    "pod_ip": "10.0.0.1",
                    "consanguinity": "",
                }
            ],
            "ps": [],
            "chief": [],
            "evaluator": [],
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/nodes")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("worker", data)
        self.assertEqual(len(data["worker"]), 1)
        self.assertEqual(data["worker"][0]["id"], 1)
        self.assertEqual(data["worker"][0]["status"], "RUNNING")

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_nodes_empty(self, mock_get_job_context, mock_json):
        """Test retrieval of nodes when no nodes exist."""
        mock_job_ctx = Mock()
        mock_job_ctx.get_mutable_job_nodes.return_value = {}
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "worker": [],
            "ps": [],
            "chief": [],
            "evaluator": [],
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/nodes")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        for node_type in ["worker", "ps", "chief", "evaluator"]:
            self.assertIn(node_type, data)
            self.assertEqual(len(data[node_type]), 0)


class TestContextHandler(AsyncHTTPTestCase):
    """Test ContextHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.Context.singleton_instance")
    def test_get_context_success(self, mock_singleton, mock_json):
        """Test successful retrieval of context data."""
        # Mock context instance using SimpleNamespace to avoid Mock __dict__ issues
        from types import SimpleNamespace

        mock_context = SimpleNamespace(
            master_service_type="grpc",
            train_speed_record_num=50,
            auto_worker_enabled=False,
            dashboard_port=8080,
            private_attr="should not be included",
            _private_method=lambda: "should not be included",
        )
        mock_singleton.return_value = mock_context

        # Mock json.dumps to return serializable data
        expected_data = {
            "master_service_type": "grpc",
            "train_speed_record_num": 50,
            "auto_worker_enabled": False,
            "dashboard_port": 8080,
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/context")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("master_service_type", data)
        self.assertEqual(data["master_service_type"], "grpc")
        self.assertIn("train_speed_record_num", data)
        self.assertEqual(data["train_speed_record_num"], 50)
        self.assertNotIn("private_attr", data)
        self.assertNotIn("_private_method", data)

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.Context.singleton_instance")
    def test_get_context_private_attrs_filtered(
        self, mock_singleton, mock_json
    ):
        """Test that private attributes are filtered out."""
        # Mock context instance using SimpleNamespace to avoid Mock __dict__ issues
        from types import SimpleNamespace

        mock_context = SimpleNamespace(
            public_attr="visible",
            _private_attr_1="invisible",
            __very_private="invisible",
            method=lambda: "should not be included",
        )
        mock_singleton.return_value = mock_context

        # Mock json.dumps to return serializable data
        expected_data = {"public_attr": "visible"}
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/context")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertEqual(len(data), 1)
        self.assertIn("public_attr", data)
        self.assertEqual(data["public_attr"], "visible")

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.Context.singleton_instance")
    def test_get_context_error_handling(self, mock_singleton, mock_json):
        """Test error handling when Context creation fails."""
        mock_singleton.side_effect = Exception("Context creation failed")

        # Mock json.dumps for error response
        mock_json.dumps.return_value = json.dumps(
            {"error": "Context creation failed"}
        )

        response = self.fetch("/api/context")
        self.assertEqual(response.code, 500)

        data = json.loads(response.body)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Context creation failed")


class TestJobArgsHandler(AsyncHTTPTestCase):
    """Test JobArgsHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_job_args_with_context(self, mock_get_job_context, mock_json):
        """Test retrieval of JobArgs when job context has job_args."""
        # Create mock job args using SimpleNamespace to avoid Mock __dict__ issues
        from types import SimpleNamespace

        # Create resource limits object
        mock_resource_limits = SimpleNamespace(cpu=4, memory=8192)

        # Create mock job args
        mock_job_args = SimpleNamespace(
            platform="kubernetes",
            namespace="default",
            job_name="test-job",
            enable_dynamic_sharding=True,
            enable_elastic_scheduling=True,
            distribution_strategy="PS",
            resource_limits=mock_resource_limits,
        )

        # Mock job context
        mock_job_ctx = Mock()
        mock_job_ctx.job_args = mock_job_args
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "platform": "kubernetes",
            "namespace": "default",
            "job_name": "test-job",
            "enable_dynamic_sharding": True,
            "enable_elastic_scheduling": True,
            "distribution_strategy": "PS",
            "resource_limits": {"cpu": 4, "memory": 8192},
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/jobargs")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("platform", data)
        self.assertEqual(data["platform"], "kubernetes")
        self.assertIn("enable_dynamic_sharding", data)
        self.assertTrue(data["enable_dynamic_sharding"])
        self.assertIn("resource_limits", data)
        # ResourceLimits should be converted to dict
        self.assertIsInstance(data["resource_limits"], dict)
        self.assertNotIn("_private_attr", data)

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_job_args_mock_data(self, mock_get_job_context, mock_json):
        """Test retrieval of JobArgs when creating mock data."""
        # Mock job context without job_args
        mock_job_ctx = Mock()
        mock_job_ctx.job_args = None
        mock_job_ctx._job_name = "mock-job"
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "platform": "kubernetes",
            "namespace": "default",
            "job_name": "mock-job",
            "enable_dynamic_sharding": True,
            "enable_elastic_scheduling": True,
            "distribution_strategy": "ParameterServerStrategy",
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/jobargs")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("platform", data)
        self.assertIn("namespace", data)
        self.assertIn("job_name", data)
        # Should use mock job name from context
        self.assertEqual(data["job_name"], "mock-job")

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_job_args_error_handling(
        self, mock_get_job_context, mock_json
    ):
        """Test error handling when JobArgs retrieval fails."""
        mock_get_job_context.side_effect = Exception("Job context error")

        # Mock json.dumps for error response
        mock_json.dumps.return_value = json.dumps(
            {"error": "Job context error"}
        )

        response = self.fetch("/api/jobargs")

        data = json.loads(response.body)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Job context error")


class TestMetricsHandler(AsyncHTTPTestCase):
    """Test MetricsHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_metrics_basic(self, mock_get_job_context, mock_json):
        """Test basic metrics retrieval."""
        # Mock a running node with resource usage
        mock_node = Mock()
        mock_node.status = "RUNNING"
        mock_node.used_resource = Mock()
        mock_node.used_resource.cpu = 2.0
        mock_node.used_resource.memory = 4096.0
        mock_node.used_resource.gpu_num = 1

        mock_job_ctx = Mock()
        mock_job_ctx.get_mutable_job_nodes.return_value = {
            "worker-1": mock_node
        }
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "timestamp": "2026-01-01T00:00:00",
            "resource_usage": {
                "total_cpu": 2.0,
                "total_memory": 4096.0,
                "total_gpu": 1,
            },
            "training_stats": {
                "running_workers": 1,
                "global_step": 0,
                "training_speed": 0.0,
            },
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/metrics")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("timestamp", data)
        self.assertIn("resource_usage", data)
        self.assertIn("training_stats", data)
        self.assertIn("total_cpu", data["resource_usage"])
        self.assertIn("total_memory", data["resource_usage"])
        self.assertIn("total_gpu", data["resource_usage"])

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_metrics_no_running_nodes(
        self, mock_get_job_context, mock_json
    ):
        """Test metrics when no nodes are running."""
        # Mock nodes with non-running status or no used_resource
        mock_node1 = Mock()
        mock_node1.status = "PENDING"
        mock_node1.used_resource = None

        mock_node2 = Mock()
        mock_node2.status = "FAILED"
        mock_node2.used_resource = None

        mock_job_ctx = Mock()
        mock_job_ctx.get_mutable_job_nodes.return_value = {
            "worker-1": mock_node1,
            "worker-2": mock_node2,
        }
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "timestamp": "2026-01-01T00:00:00",
            "resource_usage": {
                "total_cpu": 0,
                "total_memory": 0,
                "total_gpu": 0,
            },
            "training_stats": {
                "running_workers": 0,
                "global_step": 0,
                "training_speed": 0.0,
            },
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/metrics")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertEqual(data["resource_usage"]["total_cpu"], 0)
        self.assertEqual(data["resource_usage"]["total_memory"], 0)
        self.assertEqual(data["resource_usage"]["total_gpu"], 0)


class TestDiagnosisHandler(AsyncHTTPTestCase):
    """Test DiagnosisHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_diagnosis_basic(self, mock_get_job_context, mock_json):
        """Test basic diagnosis retrieval."""
        mock_job_ctx = Mock()
        mock_job_ctx.get_job_restart_count.return_value = 3
        mock_job_ctx.get_failed_node_cnt.return_value = 2

        # Create a failed node
        mock_node = Mock()
        mock_node.status = "FAILED"
        mock_node.type = "WORKER"
        mock_node.name = "worker-1"
        mock_node.exit_reason = "Container crashed"
        mock_node.finish_time = None

        mock_job_ctx.get_mutable_job_nodes.return_value = {
            "worker-1": mock_node
        }
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "diagnosis_results": [
                {
                    "type": "warning",
                    "description": "Diagnosis system not fully integrated",
                    "timestamp": "2026-01-01T00:00:00",
                }
            ],
            "fault_tolerance": {
                "total_restarts": 3,
                "failed_nodes": 2,
                "recent_failures": [],
            },
            "timestamp": "2026-01-01T00:00:00",
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/diagnosis")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("diagnosis_results", data)
        self.assertIn("fault_tolerance", data)
        self.assertEqual(data["fault_tolerance"]["total_restarts"], 3)
        self.assertEqual(data["fault_tolerance"]["failed_nodes"], 2)

    @patch("dlrover.dashboard.app.json")
    @patch("dlrover.dashboard.app.get_job_context")
    def test_get_diagnosis_no_failures(self, mock_get_job_context, mock_json):
        """Test diagnosis when there are no failures."""
        mock_job_ctx = Mock()
        mock_job_ctx.get_job_restart_count.return_value = 0
        mock_job_ctx.get_failed_node_cnt.return_value = 0
        mock_job_ctx.get_mutable_job_nodes.return_value = {}
        mock_get_job_context.return_value = mock_job_ctx

        # Mock json.dumps to return serializable data
        expected_data = {
            "diagnosis_results": [
                {
                    "type": "warning",
                    "description": "Diagnosis system not fully integrated",
                    "timestamp": "2026-01-01T00:00:00",
                }
            ],
            "fault_tolerance": {
                "total_restarts": 0,
                "failed_nodes": 0,
                "recent_failures": [],
            },
            "timestamp": "2026-01-01T00:00:00",
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/diagnosis")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertEqual(data["fault_tolerance"]["total_restarts"], 0)
        self.assertEqual(data["fault_tolerance"]["failed_nodes"], 0)
        self.assertEqual(len(data["fault_tolerance"]["recent_failures"]), 0)


class TestLogsHandler(AsyncHTTPTestCase):
    """Test LogsHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.json")
    @patch("os.path.exists")
    @patch("builtins.open")
    @patch.dict("os.environ", {"DLROVER_LOG_DIR": "/tmp/test_logs"})
    def test_get_logs_success(self, mock_open, mock_exists, mock_json):
        """Test successful retrieval of logs."""
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.readlines.return_value = ["Line 1\n", "Line 2\n", "Line 3\n"]
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock json.dumps to return serializable data
        expected_data = {
            "node_name": "test-node",
            "logs": "Line 1\nLine 2\nLine 3\n",
            "timestamp": "2026-01-01T00:00:00",
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/logs/test-node")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("node_name", data)
        self.assertEqual(data["node_name"], "test-node")
        self.assertIn("logs", data)
        self.assertIn("Line 1\nLine 2\nLine 3\n", data["logs"])


class TestNodeControlHandlers(AsyncHTTPTestCase):
    """Test RestartNodeHandler and StopNodeHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.json")
    def test_restart_node_success(self, mock_json):
        """Test successful node restart request."""
        # Mock json.dumps to return serializable data
        expected_data = {
            "status": "success",
            "node": "test-node",
            "message": "Restart request for test-node submitted successfully",
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/restart/test-node", method="POST", body="")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("node", data)
        self.assertEqual(data["node"], "test-node")

    @patch("dlrover.dashboard.app.json")
    def test_stop_node_success(self, mock_json):
        """Test successful node stop request."""
        # Mock json.dumps to return serializable data
        expected_data = {
            "status": "success",
            "node": "test-node",
            "message": "Stop request for test-node submitted successfully",
        }
        mock_json.dumps.return_value = json.dumps(expected_data)

        response = self.fetch("/api/stop/test-node", method="POST", body="")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "success")
        self.assertIn("node", data)
        self.assertEqual(data["node"], "test-node")


if __name__ == "__main__":
    unittest.main()
