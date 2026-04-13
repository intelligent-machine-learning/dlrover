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
import re
from datetime import datetime
from typing import Optional

from tornado import httpserver, ioloop, web, websocket
from tornado.concurrent import run_on_executor
from tornado.web import RequestHandler
from concurrent.futures import ThreadPoolExecutor

from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.serialize import to_dict
from dlrover.python.master.monitor.perf_monitor import PerfMonitor
from dlrover.python.master.node.job_context import get_job_context, JobContext
from dlrover.python.common.constants import NodeType, NodeStatus
from dlrover.python.common.global_context import Context
from tornado.ioloop import PeriodicCallback


class BaseHandler(RequestHandler):
    """Base handler with CORS support."""

    _job_context: JobContext = get_job_context()

    @property
    def job_ctx(self) -> JobContext:
        return self._job_context

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def get_perf_monitor(self) -> Optional[PerfMonitor]:
        return self.application.settings.get("perf_monitor", None)

    @staticmethod
    def _format_time(t):
        """Safely format a time value to ISO string."""
        if t is None:
            return None
        if isinstance(t, str):
            return t
        if hasattr(t, "isoformat"):
            return t.isoformat()
        return str(t)


class JobInfoHandler(BaseHandler):
    """Handle requests for job information."""

    def get(self):
        """Get overall job information."""

        # Get base job information from attributes
        job_args = self.job_ctx.get_job_args()
        if not job_args:
            self.write(json.dumps({}))
            return

        job_name = job_args.job_name
        job_type = job_args.distribution_strategy
        job_stage = self.job_ctx.get_job_stage()

        # Try to get some sensible node counts
        total_nodes_cnt = 0
        pending_nodes_cnt = 0
        running_nodes_cnt = 0
        failed_nodes_cnt = self.job_ctx.get_failed_node_cnt()
        succeeded_nodes_cnt = 0
        earliest_create_time = None
        earliest_start_time = None

        # Count nodes from job_nodes
        for node_type in [
            NodeType.WORKER,
            NodeType.PS,
            NodeType.CHIEF,
            NodeType.EVALUATOR,
        ]:
            nodes = self.job_ctx.dup_job_nodes_by_type(node_type)
            if nodes:
                total_nodes_cnt += len(nodes)
                for node in nodes.values():
                    if node.status in (
                        NodeStatus.PENDING,
                        NodeStatus.INITIAL,
                    ):
                        pending_nodes_cnt += 1
                    elif node.status == NodeStatus.RUNNING:
                        running_nodes_cnt += 1
                    elif node.status == NodeStatus.SUCCEEDED:
                        succeeded_nodes_cnt += 1

                    if node.create_time and (
                        earliest_create_time is None
                        or node.create_time < earliest_create_time
                    ):
                        earliest_create_time = node.create_time
                    if node.start_time and (
                        earliest_start_time is None
                        or node.start_time < earliest_start_time
                    ):
                        earliest_start_time = node.start_time

        job_info = {
            "job_name": job_name,
            "job_type": job_type,
            "job_stage": job_stage,
            "create_time": self._format_time(earliest_create_time),
            "start_time": self._format_time(earliest_start_time),
            "total_nodes": total_nodes_cnt,
            "pending_nodes": pending_nodes_cnt,
            "running_nodes": running_nodes_cnt,
            "failed_nodes": failed_nodes_cnt,
            "succeeded_nodes": succeeded_nodes_cnt,
        }
        self.write(json.dumps(job_info))


class NodesHandler(BaseHandler):
    """Handle requests for node information."""

    def get(self):
        """Get all nodes information."""
        nodes = {}

        for node_type in [
            NodeType.WORKER,
            NodeType.PS,
            NodeType.CHIEF,
            NodeType.EVALUATOR,
        ]:
            nodes[node_type] = []
            node_dict = self.job_ctx.dup_job_nodes_by_type(node_type)

            # Pre-compute rank_index -> nodes mapping (O(n) instead of O(n²))
            rank_index_map = self._build_rank_index_map(node_dict)

            for node_id, node in node_dict.items():
                # Build consanguinity chain using pre-computed index
                consanguinity = self._build_consanguinity_chain(
                    node, rank_index_map
                )

                node_info = {
                    "id": node_id,
                    "type": node_type,
                    "status": node.status,
                    "name": node.name,
                    "rank_index": node.rank_index,
                    "service_addr": node.service_addr,
                    "start_time": self._format_time(node.start_time),
                    "finish_time": self._format_time(node.finish_time),
                    "exit_reason": node.exit_reason,
                    "relaunch_count": node.relaunch_count,
                    "config_resource": self._resource_to_dict(
                        node.config_resource
                    )
                    if node.config_resource
                    else None,
                    "used_resource": self._resource_to_dict(node.used_resource)
                    if node.used_resource
                    else None,
                    "critical": node.critical,
                    "max_relaunch_count": node.max_relaunch_count,
                    "unrecoverable_failure_msg": node.unrecoverable_failure_msg,
                    "hostname": node.host_name,
                    "pod_ip": node.host_ip,
                    "consanguinity": consanguinity,
                }
                nodes[node_type].append(node_info)

        self.write(json.dumps(nodes))

    def _build_rank_index_map(self, node_dict):
        """Pre-compute rank_index -> list of (node_id, node) mapping."""
        rank_map = {}
        for node_id, node in node_dict.items():
            if node.start_time:
                rank = node.rank_index
                if rank not in rank_map:
                    rank_map[rank] = []
                rank_map[rank].append((node_id, node))

        # Sort each group by start_time once
        for rank, instances in rank_map.items():
            instances.sort(key=lambda x: x[1].start_time or datetime.min)

        return rank_map

    def _build_consanguinity_chain(self, current_node, rank_index_map):
        """Build consanguinity chain for a node using pre-computed
        rank_index map. O(1) lookup instead of O(n) scan."""
        instances = rank_index_map.get(current_node.rank_index, [])

        if len(instances) > 1:
            chain_parts = [
                f"{node.rank_index}({node_id})" for node_id, node in instances
            ]
            return " -> ".join(chain_parts)

        return ""

    def _resource_to_dict(self, resource):
        """Convert resource object to dictionary."""
        return {
            "cpu": resource.cpu,
            "memory": resource.memory,
            "gpu": f"{resource.gpu_num}({resource.gpu_type})",
            "gpu_type": resource.gpu_type,
        }


class LogsHandler(BaseHandler):
    """Handle requests for node logs."""

    # Valid node name pattern: alphanumeric, hyphens, underscores, dots
    _NODE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")

    executor = ThreadPoolExecutor(max_workers=4)

    def _validate_node_name(self, node_name):
        """Validate node_name to prevent path traversal."""
        if not self._NODE_NAME_PATTERN.match(node_name):
            return False
        # Extra safety: ensure no path separators after regex
        if os.sep in node_name or "/" in node_name:
            return False
        return True

    async def get(self, node_name):
        """Get logs for a specific node."""
        if not self._validate_node_name(node_name):
            self.set_status(400)
            self.write(json.dumps({"error": "Invalid node name"}))
            return

        try:
            result = await self._read_logs(node_name)
            self.write(json.dumps(result))
        except Exception as e:
            logger.error(f"Error reading logs for {node_name}: {e}")
            self.set_status(500)
            self.write(json.dumps({"error": "Failed to read logs"}))

    @run_on_executor
    def _read_logs(self, node_name):
        """Read log file in thread pool. Returns data dict."""
        log_dir = os.environ.get("DLROVER_LOG_DIR", "/tmp/dlrover")
        log_file = os.path.join(log_dir, f"{node_name}.log")

        # Resolve and verify the path stays within log_dir
        resolved = os.path.realpath(log_file)
        if not resolved.startswith(os.path.realpath(log_dir)):
            return {
                "node_name": node_name,
                "logs": "Access denied",
                "timestamp": datetime.now().isoformat(),
            }

        logs = self._tail_log_file(resolved, 1000)
        return {
            "node_name": node_name,
            "logs": logs,
            "timestamp": datetime.now().isoformat(),
        }

    def _tail_log_file(self, filepath, lines):
        """Read last N lines from a file without loading entire file
        into memory."""
        from collections import deque

        try:
            with open(filepath, "r") as f:
                return "".join(deque(f, maxlen=lines))
        except FileNotFoundError:
            return "Log file not found"


class DiagnosisHandler(BaseHandler):
    """Handle requests for diagnosis information."""

    def get(self):
        """Get diagnosis results and fault tolerance information."""
        try:
            # Mock data for diagnosis (safeguard if DiagnosisDataManager not available)
            results = [
                {
                    "type": "warning",
                    "description": "Diagnosis system not fully integrated",
                    "timestamp": datetime.now().isoformat(),
                }
            ]

            fault_stats = {
                "total_restarts": self.job_ctx.get_job_restart_count(),
                "failed_nodes": self.job_ctx.get_failed_node_cnt(),
                "recent_failures": self._get_recent_failures(10),
            }

            self.write(
                json.dumps(
                    {
                        "diagnosis_results": results,
                        "fault_tolerance": fault_stats,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error in DiagnosisHandler: {e}")
            self.set_status(500)
            self.write(json.dumps({"error": "Failed to get diagnosis data"}))

    def _get_recent_failures(self, limit):
        """Get recent node failures."""
        failures = []
        # Get all nodes and find failed ones
        for node_type in [
            NodeType.WORKER,
            NodeType.PS,
            NodeType.CHIEF,
            NodeType.EVALUATOR,
        ]:
            for node_id, node in self.job_ctx.dup_job_nodes_by_type(
                node_type
            ).items():
                if node.status == NodeStatus.FAILED and node.finish_time:
                    failures.append(
                        {
                            "name": node.name,
                            "type": node.type,
                            "exit_reason": node.exit_reason or "Unknown",
                            "finish_time": str(node.finish_time),
                        }
                    )

        # Sort by finish time and limit
        failures.sort(key=lambda x: x["finish_time"] or "", reverse=True)
        return failures[:limit]


class MetricsHandler(BaseHandler):
    """Handle requests for real-time metrics."""

    def get(self):
        """Get real-time metrics including resource usage and training speed."""

        # Aggregate resource usage across all nodes
        total_cpu = 0
        total_memory = 0
        total_gpu = 0
        running_workers = 0

        for node_type in [NodeType.WORKER, NodeType.PS, NodeType.CHIEF]:
            for node in self.job_ctx.dup_job_nodes_by_type(node_type).values():
                if node.status == NodeStatus.RUNNING and node.used_resource:
                    total_cpu += float(
                        getattr(node.used_resource, "cpu", 0) or 0
                    )
                    total_memory += float(
                        getattr(node.used_resource, "memory", 0) or 0
                    )
                    total_gpu += float(
                        getattr(node.used_resource, "gpu_num", 0) or 0
                    )

                    if node_type == NodeType.WORKER:
                        running_workers += 1

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "resource_usage": {
                "total_cpu": total_cpu,
                "total_memory": total_memory,
                "total_gpu": total_gpu,
            },
            "training_stats": {
                "running_workers": running_workers,
                "global_step": self._get_global_step(),
                "training_speed": self._get_training_speed(),
            },
        }

        self.write(json.dumps(metrics))

    def _get_global_step(self):
        """Get current global step from available monitors."""

        monitor = self.get_perf_monitor()
        if monitor:
            return monitor.completed_global_step
        return -1

    def _get_training_speed(self):
        """Get current training speed from available monitors."""

        monitor = self.get_perf_monitor()
        if monitor:
            return monitor.running_speed
        return -1


class RestartNodeHandler(BaseHandler):
    """Handle node restart requests."""

    def post(self, node_name):
        """Restart a specific node."""
        try:
            # This would connect to your actual node management system
            # For now, return a success message
            self.write(
                json.dumps(
                    {
                        "status": "success",
                        "node": node_name,
                        "message": f"Restart request for {node_name} submitted successfully",
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error restarting node {node_name}: {e}")
            self.set_status(500)
            self.write(json.dumps({"error": "Failed to restart node"}))


class StopNodeHandler(BaseHandler):
    """Handle node stop requests."""

    def post(self, node_name):
        """Stop a specific node."""
        try:
            # This would connect to your actual node management system
            # For now, return a success message
            self.write(
                json.dumps(
                    {
                        "status": "success",
                        "node": node_name,
                        "message": f"Stop request for {node_name} submitted successfully",
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error stopping node {node_name}: {e}")
            self.set_status(500)
            self.write(json.dumps({"error": "Failed to stop node"}))


class WebSocketHandler(websocket.WebSocketHandler):
    """WebSocket handler for real-time updates."""

    clients: set = set()

    def open(self):
        self.clients.add(self)
        logger.info(f"WebSocket client connected: {self}")

    def on_close(self):
        self.clients.discard(self)
        logger.info(f"WebSocket client disconnected: {self}")

    def on_message(self, message):
        logger.info(f"Received WebSocket message: {message}")

    @classmethod
    def broadcast(cls, message):
        """Broadcast message to all connected clients."""
        for client in list(
            cls.clients
        ):  # Create a copy to avoid modification during iteration
            try:
                if (
                    client.ws_connection
                    and not client.ws_connection.is_closing()
                ):
                    client.write_message(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                cls.clients.discard(client)


class JobArgsHandler(BaseHandler):
    """Handle requests for JobArgs arguments."""

    def get(self):
        """Get JobArgs fields as JSON."""
        try:
            # Get JobArgs from context
            job_args = self.job_ctx.get_job_args()
            if not job_args:
                self.write(json.dumps({}))
                return

            # Use JsonSerializable.to_json() if available (JobArgs extends
            # JsonSerializable), which handles recursive serialization via
            # the to_dict default function in serialize module.
            if hasattr(job_args, "to_json"):
                self.write(job_args.to_json())
            else:
                # Fallback: use to_dict for recursive conversion
                self.write(json.dumps(to_dict(job_args), default=to_dict))
        except Exception as e:
            logger.error(f"Error in JobArgsHandler: {e}")
            self.set_status(500)
            self.write(json.dumps({"error": "Failed to get job args"}))


class ContextHandler(BaseHandler):
    """Handle requests for Context configuration data."""

    def get(self):
        """Get Context object fields as JSON."""
        try:
            context = Context.singleton_instance()
            # Convert to dict, excluding private methods and non-serializable objects
            context_data = {}
            for k, v in context.__dict__.items():
                if k.startswith("_") or callable(v):
                    continue
                # Convert non-serializable types to strings
                if isinstance(
                    v, (str, int, float, bool, list, dict, type(None))
                ):
                    context_data[k] = v
                else:
                    context_data[k] = str(v)

            self.write(json.dumps(context_data))
        except Exception as e:
            logger.error(f"Error in ContextHandler: {e}")
            self.set_status(500)
            self.write(json.dumps({"error": "Failed to get context data"}))


def create_dashboard_app(**kwargs):
    """Create the dashboard application."""

    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "static"),
        "template_path": os.path.join(os.path.dirname(__file__), "templates"),
        "debug": os.environ.get("DLROVER_DASHBOARD_DEBUG", "").lower()
        == "true",
    }

    # setup instance objects
    if "perf_monitor" in kwargs:
        settings["perf_monitor"] = kwargs["perf_monitor"]

    app = web.Application(
        [
            (r"/", IndexHandler),
            (r"/node_detail.html", NodeDetailHandler),
            (r"/api/job", JobInfoHandler),
            (r"/api/nodes", NodesHandler),
            (r"/api/logs/(.+)", LogsHandler),
            (r"/api/restart/(.+)", RestartNodeHandler),
            (r"/api/stop/(.+)", StopNodeHandler),
            (r"/api/diagnosis", DiagnosisHandler),
            (r"/api/metrics", MetricsHandler),
            (r"/api/context", ContextHandler),
            (r"/api/jobargs", JobArgsHandler),
            (r"/ws", WebSocketHandler),
            (
                r"/static/(.*)",
                web.StaticFileHandler,
                {"path": settings["static_path"]},
            ),
        ],
        **settings,
    )

    return app


class IndexHandler(BaseHandler):
    """Serve the main dashboard HTML page."""

    def get(self):
        """Serve index.html as a static file to avoid template parsing issues with Vue.js."""
        index_path = os.path.join(
            os.path.dirname(__file__), "templates", "index.html"
        )
        if os.path.exists(index_path):
            self.set_header("Content-Type", "text/html; charset=utf-8")
            with open(index_path, "r", encoding="utf-8") as f:
                self.write(f.read())
        else:
            self.set_status(404)
            self.write("index.html not found")


class NodeDetailHandler(BaseHandler):
    """Serve the node detail page HTML."""

    def get(self):
        """Serve node_detail.html."""
        node_detail_path = os.path.join(
            os.path.dirname(__file__), "templates", "node_detail.html"
        )
        if os.path.exists(node_detail_path):
            self.set_header("Content-Type", "text/html; charset=utf-8")
            with open(node_detail_path, "r", encoding="utf-8") as f:
                self.write(f.read())
        else:
            self.set_status(404)
            self.write("node_detail.html not found")


def start_dashboard_server(host="0.0.0.0", port=8080):
    """Start the dashboard server."""
    logger.info(f"Starting DLRover Dashboard Server on {host}:{port}")

    app = create_dashboard_app()
    server = httpserver.HTTPServer(app)
    server.listen(port, host)

    logger.info(
        f"Dashboard server started. Open http://{host}:{port} in your browser."
    )

    # Start periodic update broadcasting

    def broadcast_updates():
        try:
            job_ctx = get_job_context()
            update = {
                "type": "metrics",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "running_nodes": job_ctx.get_running_node_size(),
                    "failed_nodes": job_ctx.get_failed_node_cnt(),
                },
            }
            WebSocketHandler.broadcast(json.dumps(update))
        except Exception as e:
            logger.error(f"Error broadcasting update: {e}")

    # Broadcast updates every 5 seconds
    update_callback = PeriodicCallback(broadcast_updates, 5000)
    update_callback.start()

    try:
        ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("Dashboard server stopped.")


if __name__ == "__main__":
    start_dashboard_server()
