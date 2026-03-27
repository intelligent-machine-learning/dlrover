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
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application

# Import the module to test
from dlrover.dashboard.app import (
    BaseHandler,
    WebSocketHandler,
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


class TestFormatTime(unittest.TestCase):
    """Test BaseHandler._format_time static method."""

    def test_none(self):
        self.assertIsNone(BaseHandler._format_time(None))

    def test_string_passthrough(self):
        self.assertEqual(BaseHandler._format_time("2026-01-15"), "2026-01-15")

    def test_datetime_isoformat(self):
        dt = datetime(2026, 1, 15, 10, 30, 0)
        self.assertEqual(BaseHandler._format_time(dt), "2026-01-15T10:30:00")

    def test_fallback_to_str(self):
        """Non-string, non-datetime objects use str()."""
        self.assertEqual(BaseHandler._format_time(12345), "12345")


class TestCreateDashboardApp(unittest.TestCase):
    """Test create_dashboard_app factory."""

    def test_creates_app(self):
        app = create_dashboard_app()
        self.assertIsNotNone(app)

    def test_perf_monitor_in_settings(self):
        monitor = MagicMock()
        app = create_dashboard_app(perf_monitor=monitor)
        self.assertIs(app.settings["perf_monitor"], monitor)

    @patch.dict(os.environ, {"DLROVER_DASHBOARD_DEBUG": "true"})
    def test_debug_mode(self):
        app = create_dashboard_app()
        self.assertTrue(app.settings["debug"])


class TestIndexHandler(AsyncHTTPTestCase):
    """Test IndexHandler serving index.html."""

    def get_app(self):
        return create_dashboard_app()

    def test_get_index(self):
        response = self.fetch("/")
        self.assertEqual(response.code, 200)
        self.assertIn("text/html", response.headers["Content-Type"])

    @patch("dlrover.dashboard.app.os.path.exists", return_value=False)
    def test_get_index_missing(self, _):
        response = self.fetch("/")
        self.assertEqual(response.code, 404)


class TestNodeDetailHandler(AsyncHTTPTestCase):
    """Test NodeDetailHandler serving node_detail.html."""

    def get_app(self):
        return create_dashboard_app()

    def test_get_node_detail(self):
        response = self.fetch("/node_detail.html")
        self.assertEqual(response.code, 200)
        self.assertIn("text/html", response.headers["Content-Type"])

    @patch("dlrover.dashboard.app.os.path.exists", return_value=False)
    def test_get_node_detail_missing(self, _):
        response = self.fetch("/node_detail.html")
        self.assertEqual(response.code, 404)


class TestMetricsHandlerWithPerfMonitor(AsyncHTTPTestCase):
    """Test MetricsHandler when perf_monitor is available."""

    def get_app(self):
        self.mock_monitor = MagicMock()
        self.mock_monitor.completed_global_step = 1500
        self.mock_monitor.running_speed = 3.5
        return create_dashboard_app(perf_monitor=self.mock_monitor)

    def test_metrics_with_perf_monitor(self):
        response = self.fetch("/api/metrics")
        self.assertEqual(response.code, 200)

        data = json.loads(response.body)
        self.assertEqual(data["training_stats"]["global_step"], 1500)
        self.assertEqual(data["training_stats"]["training_speed"], 3.5)


class TestMetricsHandlerNoPerfMonitor(AsyncHTTPTestCase):
    """Test MetricsHandler when no perf_monitor is set."""

    def get_app(self):
        return create_dashboard_app()

    def test_metrics_no_perf_monitor_returns_negative(self):
        """Without perf_monitor, global_step and speed are -1."""
        from dlrover.python.master.node.job_context import get_job_context

        get_job_context().clear_job_nodes()

        response = self.fetch("/api/metrics")
        data = json.loads(response.body)
        self.assertEqual(data["training_stats"]["global_step"], -1)
        self.assertEqual(data["training_stats"]["training_speed"], -1)


class TestLogsHandlerDirect(AsyncHTTPTestCase):
    """Test LogsHandler with real file operations."""

    def get_app(self):
        return create_dashboard_app()

    def test_read_actual_log_file(self):
        """Test reading an actual log file from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a test log file
            log_file = os.path.join(tmpdir, "test-node.log")
            with open(log_file, "w") as f:
                f.write("line1\nline2\nline3\n")

            with patch.dict(os.environ, {"DLROVER_LOG_DIR": tmpdir}):
                response = self.fetch("/api/logs/test-node")
                self.assertEqual(response.code, 200)
                data = json.loads(response.body)
                self.assertEqual(data["node_name"], "test-node")
                self.assertIn("line1", data["logs"])
                self.assertIn("line3", data["logs"])

    def test_read_missing_log_file(self):
        """Test reading a log file that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"DLROVER_LOG_DIR": tmpdir}):
                response = self.fetch("/api/logs/nonexistent-node")
                self.assertEqual(response.code, 200)
                data = json.loads(response.body)
                self.assertIn("not found", data["logs"])

    def test_path_traversal_via_symlink(self):
        """Test that symlink-based traversal is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # This is tested indirectly — the node name regex already
            # blocks slashes, but the realpath check adds defense in depth
            with patch.dict(os.environ, {"DLROVER_LOG_DIR": tmpdir}):
                response = self.fetch("/api/logs/..%2F..%2Fetc%2Fpasswd")
                self.assertEqual(response.code, 400)

    def test_valid_node_name_patterns(self):
        """Test that valid node name patterns are accepted."""
        from dlrover.dashboard.app import LogsHandler

        handler = LogsHandler
        self.assertTrue(handler._NODE_NAME_PATTERN.match("worker-0"))
        self.assertTrue(handler._NODE_NAME_PATTERN.match("ps_1.pod"))
        self.assertTrue(handler._NODE_NAME_PATTERN.match("chief-0"))
        self.assertIsNone(handler._NODE_NAME_PATTERN.match("no spaces"))
        self.assertIsNone(handler._NODE_NAME_PATTERN.match("a/b"))


class TestJobArgsHandlerFallback(AsyncHTTPTestCase):
    """Test JobArgsHandler fallback path (no to_json method)."""

    def get_app(self):
        return create_dashboard_app()

    def test_fallback_without_to_json(self):
        """Test fallback when job_args has no to_json method."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()

        # Create a simple object without to_json
        class SimpleArgs:
            def __init__(self):
                self.platform = "test"
                self.job_name = "fallback-job"

        job_ctx._job_args = SimpleArgs()

        try:
            response = self.fetch("/api/jobargs")
            self.assertEqual(response.code, 200)
            data = json.loads(response.body)
            self.assertEqual(data["platform"], "test")
            self.assertEqual(data["job_name"], "fallback-job")
        finally:
            job_ctx._job_args = None

    @patch(
        "dlrover.dashboard.app.BaseHandler.job_ctx",
        new_callable=lambda: property(
            lambda self: MagicMock(
                get_job_args=MagicMock(side_effect=RuntimeError("err"))
            )
        ),
    )
    def test_job_args_exception(self, _):
        """Test error handling in JobArgsHandler."""
        # Patch at handler level to force exception
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        original = job_ctx.get_job_args

        def raise_err():
            raise RuntimeError("err")

        job_ctx.get_job_args = raise_err
        try:
            response = self.fetch("/api/jobargs")
            self.assertEqual(response.code, 500)
            data = json.loads(response.body)
            self.assertIn("error", data)
        finally:
            job_ctx.get_job_args = original


class TestWebSocketHandler(unittest.TestCase):
    """Test WebSocketHandler class methods."""

    def setUp(self):
        # Reset clients set between tests
        WebSocketHandler.clients = set()

    def test_broadcast_no_clients(self):
        """Broadcast with no connected clients should not raise."""
        WebSocketHandler.broadcast('{"test": true}')

    def test_broadcast_to_client(self):
        mock_client = MagicMock()
        mock_client.ws_connection = MagicMock()
        mock_client.ws_connection.is_closing.return_value = False
        WebSocketHandler.clients.add(mock_client)

        WebSocketHandler.broadcast('{"type": "update"}')

        mock_client.write_message.assert_called_once_with('{"type": "update"}')

    def test_broadcast_skips_closing_client(self):
        mock_client = MagicMock()
        mock_client.ws_connection = MagicMock()
        mock_client.ws_connection.is_closing.return_value = True
        WebSocketHandler.clients.add(mock_client)

        WebSocketHandler.broadcast('{"type": "update"}')

        mock_client.write_message.assert_not_called()

    def test_broadcast_skips_none_ws_connection(self):
        mock_client = MagicMock()
        mock_client.ws_connection = None
        WebSocketHandler.clients.add(mock_client)

        WebSocketHandler.broadcast('{"type": "update"}')

        mock_client.write_message.assert_not_called()

    def test_broadcast_removes_erroring_client(self):
        mock_client = MagicMock()
        mock_client.ws_connection = MagicMock()
        mock_client.ws_connection.is_closing.return_value = False
        mock_client.write_message.side_effect = RuntimeError("send err")
        WebSocketHandler.clients.add(mock_client)

        WebSocketHandler.broadcast('{"type": "update"}')

        self.assertNotIn(mock_client, WebSocketHandler.clients)

    def _make_handler(self):
        """Create a mock handler that shares the class-level clients set."""
        handler = MagicMock()
        handler.clients = WebSocketHandler.clients
        return handler

    def test_open_adds_client(self):
        handler = self._make_handler()
        WebSocketHandler.open(handler)
        self.assertIn(handler, WebSocketHandler.clients)

    def test_on_close_removes_client(self):
        handler = self._make_handler()
        WebSocketHandler.clients.add(handler)
        WebSocketHandler.on_close(handler)
        self.assertNotIn(handler, WebSocketHandler.clients)

    def test_on_close_absent_client_no_error(self):
        handler = self._make_handler()
        # Not in clients set — discard should not raise
        WebSocketHandler.on_close(handler)

    def test_on_message_logs(self):
        handler = self._make_handler()
        with patch("dlrover.dashboard.app.logger") as mock_logger:
            WebSocketHandler.on_message(handler, "hello")
            mock_logger.info.assert_called()


class TestDiagnosisHandlerRecentFailures(AsyncHTTPTestCase):
    """Test DiagnosisHandler with multiple failure types."""

    def get_app(self):
        return create_dashboard_app()

    def test_failures_sorted_by_time_and_limited(self):
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        # Create 3 failed nodes with different finish times
        for i in range(3):
            node = Node(
                NodeType.WORKER,
                i,
                name=f"worker-{i}",
                status=NodeStatus.FAILED,
            )
            node.finish_time = datetime(2026, 1, 15, 10 + i, 0, 0)
            node.exit_reason = f"Error-{i}"
            job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/diagnosis")
            self.assertEqual(response.code, 200)
            data = json.loads(response.body)
            failures = data["fault_tolerance"]["recent_failures"]
            self.assertEqual(len(failures), 3)
            # Should be sorted by finish_time descending
            self.assertEqual(failures[0]["name"], "worker-2")
            self.assertEqual(failures[-1]["name"], "worker-0")
        finally:
            job_ctx.clear_job_nodes()

    def test_failure_without_exit_reason(self):
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        node = Node(
            NodeType.PS,
            0,
            name="ps-0",
            status=NodeStatus.FAILED,
        )
        node.finish_time = datetime(2026, 1, 15, 10, 0, 0)
        # No exit_reason set — should default to "Unknown"
        job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/diagnosis")
            data = json.loads(response.body)
            failures = data["fault_tolerance"]["recent_failures"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["exit_reason"], "Unknown")
        finally:
            job_ctx.clear_job_nodes()


class TestNodesHandlerResourceConversion(AsyncHTTPTestCase):
    """Test NodesHandler._resource_to_dict and edge cases."""

    def get_app(self):
        return create_dashboard_app()

    def test_node_without_used_resource(self):
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        node = Node(
            NodeType.WORKER,
            0,
            name="worker-0",
            status=NodeStatus.RUNNING,
            start_time=datetime(2026, 1, 15, 10, 0, 0),
        )
        # config_resource and used_resource are both None
        node.config_resource = None
        node.used_resource = None
        job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/nodes")
            data = json.loads(response.body)
            w = data[NodeType.WORKER][0]
            self.assertIsNone(w["config_resource"])
            self.assertIsNone(w["used_resource"])
        finally:
            job_ctx.clear_job_nodes()

    def test_node_without_start_time_excluded_from_rank_map(self):
        """Nodes without start_time should not appear in rank_index_map."""
        from dlrover.python.master.node.job_context import get_job_context

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()

        node = Node(
            NodeType.WORKER,
            0,
            name="worker-0",
            status=NodeStatus.PENDING,
            rank_index=0,
        )
        # No start_time
        job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/nodes")
            data = json.loads(response.body)
            w = data[NodeType.WORKER][0]
            self.assertEqual(w["consanguinity"], "")
        finally:
            job_ctx.clear_job_nodes()


class TestJobInfoHandlerSucceededNodes(AsyncHTTPTestCase):
    """Test JobInfoHandler with succeeded nodes."""

    def get_app(self):
        return create_dashboard_app()

    def test_succeeded_nodes_counted(self):
        from dlrover.python.master.node.job_context import get_job_context
        from dlrover.python.scheduler.job import LocalJobArgs

        job_ctx = get_job_context()
        job_ctx.clear_job_nodes()
        job_args = LocalJobArgs("local", "default", "test")
        job_ctx.set_job_args(job_args)

        node = Node(
            NodeType.WORKER,
            0,
            name="worker-0",
            status=NodeStatus.SUCCEEDED,
        )
        job_ctx.update_job_node(node)

        try:
            response = self.fetch("/api/job")
            data = json.loads(response.body)
            self.assertEqual(data["succeeded_nodes"], 1)
        finally:
            job_ctx.clear_job_nodes()
            job_ctx._job_args = None


if __name__ == "__main__":
    unittest.main()
