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
from datetime import datetime
from unittest.mock import patch
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application

# Import the module to test
from dlrover.dashboard.app import (
    BaseHandler,
    create_dashboard_app,
)
from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.node import Node, NodeResource


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

    def test_get_job_info_no_job_args(self):
        """Test job info returns empty when no job_args set."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        # Ensure no job_args set
        job_ctx._job_args = None

        response = self.fetch("/api/job")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data, {})

    def test_get_job_info_with_nodes(self):
        """Test job info with actual nodes in job context."""
        from dlrover.python.master.node.job_context import get_job_context
        from dlrover.python.scheduler.job import LocalJobArgs

        job_ctx = get_job_context()

        # Set up job args
        job_args = LocalJobArgs("local", "default", "test-job")
        job_ctx.set_job_args(job_args)

        # Set up nodes
        now = datetime(2026, 1, 15, 10, 0, 0)
        node0 = Node(
            NodeType.WORKER,
            0,
            name="worker-0",
            status=NodeStatus.RUNNING,
            start_time=now,
        )
        node0.create_time = now
        node1 = Node(
            NodeType.WORKER,
            1,
            name="worker-1",
            status=NodeStatus.PENDING,
        )
        job_ctx.update_job_node(node0)
        job_ctx.update_job_node(node1)

        try:
            response = self.fetch("/api/job")
            self.assertEqual(response.code, 200)

            data = json.loads(response.body)
            self.assertEqual(data["job_name"], "test-job")
            self.assertIn("job_stage", data)
            self.assertEqual(data["total_nodes"], 2)
            self.assertEqual(data["running_nodes"], 1)
            self.assertEqual(data["pending_nodes"], 1)
            # create_time/start_time should come from actual node data
            self.assertEqual(data["create_time"], now.isoformat())
            self.assertEqual(data["start_time"], now.isoformat())
        finally:
            # Cleanup
            job_ctx.clear_job_nodes()
            job_ctx._job_args = None

    def test_get_job_info_no_nodes_times_are_none(self):
        """Test that create_time/start_time are None when no nodes exist."""
        from dlrover.python.master.node.job_context import get_job_context
        from dlrover.python.scheduler.job import LocalJobArgs

        job_ctx = get_job_context()
        job_args = LocalJobArgs("local", "default", "empty-job")
        job_ctx.set_job_args(job_args)

        try:
            response = self.fetch("/api/job")
            self.assertEqual(response.code, 200)

            data = json.loads(response.body)
            self.assertIsNone(data["create_time"])
            self.assertIsNone(data["start_time"])
            self.assertEqual(data["total_nodes"], 0)
        finally:
            job_ctx.clear_job_nodes()
            job_ctx._job_args = None


class TestNodesHandler(AsyncHTTPTestCase):
    """Test NodesHandler."""

    def get_app(self):
        return create_dashboard_app()

    def test_get_nodes_empty(self):
        """Test retrieval of nodes when no nodes exist."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        response = self.fetch("/api/nodes")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        for node_type in [
            NodeType.WORKER,
            NodeType.PS,
            NodeType.CHIEF,
            NodeType.EVALUATOR,
        ]:
            self.assertIn(node_type, data)
            self.assertEqual(len(data[node_type]), 0)

    def test_get_nodes_with_data(self):
        """Test retrieval of nodes with actual node data."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        now = datetime(2026, 1, 15, 10, 0, 0)
        node = Node(
            NodeType.WORKER,
            0,
            config_resource=NodeResource(4.0, 8192),
            name="worker-0",
            status=NodeStatus.RUNNING,
            start_time=now,
            rank_index=0,
            host_name="host-0",
            host_ip="10.0.0.1",
        )
        job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/nodes")
            self.assertEqual(response.code, 200)

            data = json.loads(response.body)
            workers = data[NodeType.WORKER]
            self.assertEqual(len(workers), 1)

            w = workers[0]
            self.assertEqual(w["id"], 0)
            self.assertEqual(w["status"], NodeStatus.RUNNING)
            self.assertEqual(w["name"], "worker-0")
            self.assertEqual(w["rank_index"], 0)
            self.assertEqual(w["start_time"], now.isoformat())
            self.assertEqual(w["hostname"], "host-0")
            self.assertEqual(w["pod_ip"], "10.0.0.1")
            self.assertIsNotNone(w["config_resource"])
            self.assertEqual(w["config_resource"]["cpu"], 4.0)
            self.assertEqual(w["config_resource"]["memory"], 8192)
            # Single node — no consanguinity chain
            self.assertEqual(w["consanguinity"], "")
        finally:
            job_ctx.clear_job_nodes()

    def test_get_nodes_consanguinity_chain(self):
        """Test consanguinity chain is built for relaunched nodes."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        t1 = datetime(2026, 1, 15, 10, 0, 0)
        t2 = datetime(2026, 1, 15, 11, 0, 0)

        # Two physical nodes with the same rank_index (relaunch)
        node0 = Node(
            NodeType.WORKER,
            0,
            name="worker-0",
            status=NodeStatus.FAILED,
            start_time=t1,
            rank_index=0,
        )
        node1 = Node(
            NodeType.WORKER,
            1,
            name="worker-0-relaunch",
            status=NodeStatus.RUNNING,
            start_time=t2,
            rank_index=0,
        )
        job_ctx.update_job_node(node0)
        job_ctx.update_job_node(node1)

        try:
            response = self.fetch("/api/nodes")
            self.assertEqual(response.code, 200)

            data = json.loads(response.body)
            workers = data[NodeType.WORKER]
            self.assertEqual(len(workers), 2)

            # Both should have the same consanguinity chain
            for w in workers:
                self.assertIn("->", w["consanguinity"])
                self.assertIn("0(0)", w["consanguinity"])
                self.assertIn("0(1)", w["consanguinity"])
        finally:
            job_ctx.clear_job_nodes()

    def test_get_nodes_start_time_string(self):
        """Test nodes with string start_time are handled correctly."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        node = Node(
            NodeType.WORKER,
            0,
            name="worker-0",
            status=NodeStatus.RUNNING,
            start_time="2026-01-15T10:00:00",
            rank_index=0,
        )
        job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/nodes")
            self.assertEqual(response.code, 200)

            data = json.loads(response.body)
            w = data[NodeType.WORKER][0]
            self.assertEqual(w["start_time"], "2026-01-15T10:00:00")
        finally:
            job_ctx.clear_job_nodes()


class TestContextHandler(AsyncHTTPTestCase):
    """Test ContextHandler."""

    def get_app(self):
        return create_dashboard_app()

    def test_get_context_success(self):
        """Test successful retrieval of context data."""
        response = self.fetch("/api/context")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        # Context should not contain private attributes
        for key in data:
            self.assertFalse(key.startswith("_"))

    @patch("dlrover.dashboard.app.Context.singleton_instance")
    def test_get_context_error_handling(self, mock_singleton):
        """Test error handling when Context creation fails."""
        mock_singleton.side_effect = Exception("Context creation failed")

        response = self.fetch("/api/context")
        self.assertEqual(response.code, 500)

        data = json.loads(response.body)
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Failed to get context data")


class TestJobArgsHandler(AsyncHTTPTestCase):
    """Test JobArgsHandler."""

    def get_app(self):
        return create_dashboard_app()

    def test_get_job_args_empty(self):
        """Test retrieval when no job_args set."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx._job_args = None

        response = self.fetch("/api/jobargs")
        self.assertEqual(response.code, 200)
        data = json.loads(response.body)
        self.assertEqual(data, {})

    def test_get_job_args_with_real_job_args(self):
        """Test retrieval of JobArgs with actual JobArgs object.

        This tests that nested objects (NodeArgs, NodeGroupResource,
        ResourceLimits) are properly serialized via JsonSerializable.to_json().
        """
        from dlrover.python.master.node.job_context import get_job_context
        from dlrover.python.scheduler.job import LocalJobArgs

        job_ctx = get_job_context()
        job_args = LocalJobArgs("local", "default", "test-job")
        job_ctx.set_job_args(job_args)

        try:
            response = self.fetch("/api/jobargs")
            self.assertEqual(response.code, 200)

            data = json.loads(response.body)
            self.assertEqual(data["platform"], "local")
            self.assertEqual(data["namespace"], "default")
            self.assertEqual(data["job_name"], "test-job")
            self.assertIn("distribution_strategy", data)
            # resource_limits should be recursively serialized
            self.assertIn("resource_limits", data)
            rl = data["resource_limits"]
            self.assertIn("cpu", rl)
            self.assertIn("memory", rl)
        finally:
            job_ctx._job_args = None

    def test_get_job_args_with_node_args(self):
        """Test that node_args with nested objects serialize correctly."""
        from dlrover.python.master.node.job_context import get_job_context
        from dlrover.python.scheduler.job import LocalJobArgs
        from dlrover.python.common.node import NodeGroupResource, NodeResource

        job_ctx = get_job_context()
        job_args = LocalJobArgs("local", "default", "nested-test")

        # Add node_args with nested NodeGroupResource
        from dlrover.python.scheduler.job import NodeArgs

        class ConcreteNodeArgs(NodeArgs):
            pass

        group_resource = NodeGroupResource(2, NodeResource(4.0, 8192))
        node_args = ConcreteNodeArgs(group_resource)
        job_args.node_args[NodeType.WORKER] = node_args
        job_ctx.set_job_args(job_args)

        try:
            response = self.fetch("/api/jobargs")
            self.assertEqual(response.code, 200)

            # The key test: this should NOT raise TypeError
            data = json.loads(response.body)
            self.assertIn("node_args", data)
            # node_args should be serialized as a dict, not fail
            self.assertIsInstance(data["node_args"], dict)
        finally:
            job_ctx._job_args = None


class TestMetricsHandler(AsyncHTTPTestCase):
    """Test MetricsHandler."""

    def get_app(self):
        return create_dashboard_app()

    def test_get_metrics_no_nodes(self):
        """Test metrics when no nodes are running."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        response = self.fetch("/api/metrics")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("timestamp", data)
        self.assertIn("resource_usage", data)
        self.assertIn("training_stats", data)
        self.assertEqual(data["resource_usage"]["total_cpu"], 0)
        self.assertEqual(data["resource_usage"]["total_memory"], 0)
        self.assertEqual(data["resource_usage"]["total_gpu"], 0)
        self.assertEqual(data["training_stats"]["running_workers"], 0)

    def test_get_metrics_with_running_nodes(self):
        """Test metrics with running nodes."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        node = Node(
            NodeType.WORKER,
            0,
            config_resource=NodeResource(4.0, 8192),
            name="worker-0",
            status=NodeStatus.RUNNING,
        )
        node.used_resource = NodeResource(2.0, 4096)
        job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/metrics")
            self.assertEqual(response.code, 200)

            data = json.loads(response.body)
            self.assertEqual(data["resource_usage"]["total_cpu"], 2.0)
            self.assertEqual(data["resource_usage"]["total_memory"], 4096.0)
            self.assertEqual(data["training_stats"]["running_workers"], 1)
        finally:
            job_ctx.clear_job_nodes()


class TestDiagnosisHandler(AsyncHTTPTestCase):
    """Test DiagnosisHandler."""

    def get_app(self):
        return create_dashboard_app()

    def test_get_diagnosis_basic(self):
        """Test basic diagnosis retrieval."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        response = self.fetch("/api/diagnosis")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertIn("diagnosis_results", data)
        self.assertIn("fault_tolerance", data)
        self.assertIn("timestamp", data)
        self.assertEqual(len(data["fault_tolerance"]["recent_failures"]), 0)

    def test_get_diagnosis_with_failures(self):
        """Test diagnosis with failed nodes."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        now = datetime(2026, 1, 15, 12, 0, 0)
        node = Node(
            NodeType.WORKER,
            0,
            name="worker-0",
            status=NodeStatus.FAILED,
        )
        node.finish_time = now
        node.exit_reason = "OOM"
        job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/diagnosis")
            self.assertEqual(response.code, 200)

            data = json.loads(response.body)
            failures = data["fault_tolerance"]["recent_failures"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["name"], "worker-0")
            self.assertEqual(failures[0]["exit_reason"], "OOM")
            self.assertIn("2026-01-15", failures[0]["finish_time"])
        finally:
            job_ctx.clear_job_nodes()


class TestLogsHandler(AsyncHTTPTestCase):
    """Test LogsHandler."""

    def get_app(self):
        return create_dashboard_app()

    @patch("dlrover.dashboard.app.LogsHandler._read_logs")
    def test_get_logs_success(self, mock_read_logs):
        """Test successful retrieval of logs."""
        from tornado.concurrent import Future

        future = Future()
        future.set_result(
            {
                "node_name": "test-node",
                "logs": "Line 1\nLine 2\nLine 3\n",
                "timestamp": "2026-01-15T10:00:00",
            }
        )
        mock_read_logs.return_value = future

        response = self.fetch("/api/logs/test-node")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertEqual(data["node_name"], "test-node")
        self.assertIn("Line 1", data["logs"])

    def test_get_logs_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        response = self.fetch("/api/logs/../../etc/passwd")
        self.assertEqual(response.code, 400)

        data = json.loads(response.body)
        self.assertEqual(data["error"], "Invalid node name")

    def test_get_logs_invalid_node_name(self):
        """Test that invalid node names are rejected."""
        response = self.fetch("/api/logs/node%20name%20with%20spaces")
        self.assertEqual(response.code, 400)


class TestNodeControlHandlers(AsyncHTTPTestCase):
    """Test RestartNodeHandler and StopNodeHandler."""

    def get_app(self):
        return create_dashboard_app()

    def test_restart_node_success(self):
        """Test successful node restart request."""
        response = self.fetch("/api/restart/test-node", method="POST", body="")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["node"], "test-node")

    def test_stop_node_success(self):
        """Test successful node stop request."""
        response = self.fetch("/api/stop/test-node", method="POST", body="")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["node"], "test-node")


if __name__ == "__main__":
    unittest.main()
