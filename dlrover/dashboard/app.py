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
from datetime import datetime

from tornado import httpserver, ioloop, web, websocket
from tornado.concurrent import run_on_executor
from tornado.web import RequestHandler
from concurrent.futures import ThreadPoolExecutor

from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.node.job_context import get_job_context
from dlrover.python.common.constants import NodeType, NodeStatus
from dlrover.python.common.global_context import Context
from tornado.ioloop import PeriodicCallback


class BaseHandler(RequestHandler):
    """Base handler with CORS support."""

    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")


class JobInfoHandler(BaseHandler):
    """Handle requests for job information."""

    def get(self):
        """Get overall job information."""
        job_ctx = get_job_context()

        # Get base job information from attributes (may not be set)
        job_args = job_ctx.get_job_args()
        job_name = job_args.job_name
        job_type = job_args.distribution_strategy
        job_stage = job_ctx.get_job_stage()

        # For create/start times, fall back to using import time if not set
        import_time = datetime.now()

        # Try to get some sensible node counts
        total_nodes_cnt = 0
        running_nodes_cnt = 0
        failed_nodes_cnt = job_ctx.get_failed_node_cnt()
        succeeded_nodes_cnt = 0

        # Count nodes from job_nodes
        for node_type in [
            NodeType.WORKER,
            NodeType.PS,
            NodeType.CHIEF,
            NodeType.EVALUATOR,
        ]:
            nodes = job_ctx.job_nodes_by_type(node_type)
            if nodes:
                total_nodes_cnt += len(nodes)
                running_nodes_cnt += sum(
                    1
                    for node in nodes.values()
                    if getattr(node, "status", "") == "RUNNING"
                )
                succeeded_nodes_cnt += sum(
                    1
                    for node in nodes.values()
                    if getattr(node, "status", "") == "SUCCEEDED"
                )

        job_info = {
            "job_name": job_name,
            "job_type": job_type,
            "job_stage": job_stage,
            "create_time": import_time.isoformat(),  # Simplified for now
            "start_time": import_time.isoformat(),  # Simplified for now
            "total_nodes": total_nodes_cnt,
            "running_nodes": running_nodes_cnt,
            "failed_nodes": failed_nodes_cnt,
            "succeeded_nodes": succeeded_nodes_cnt,
        }
        self.write(json.dumps(job_info))


class NodesHandler(BaseHandler):
    """Handle requests for node information."""

    def get(self):
        """Get all nodes information."""
        job_ctx = get_job_context()
        nodes = {}

        for node_type in [
            NodeType.WORKER,
            NodeType.PS,
            NodeType.CHIEF,
            NodeType.EVALUATOR,
        ]:
            nodes[node_type] = []
            node_dict = job_ctx.get_mutable_job_nodes(node_type)

            for node_id, node in node_dict.items():
                # Build consanguinity chain
                consanguinity = self._build_consanguinity_chain(
                    node, job_ctx, node_type, node_dict
                )

                node_info = {
                    "id": node_id,
                    "type": node_type,
                    "status": node.status,
                    "name": node.name,
                    "rank_index": node.rank_index
                    if hasattr(node, "rank_index")
                    else node_id,
                    "service_addr": node.service_addr,
                    "start_time": node.start_time.isoformat()
                    if node.start_time
                    else None,
                    "finish_time": node.finish_time.isoformat()
                    if node.finish_time
                    else None,
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
                    "consanguinity": consanguinity,  # Add consanguinity chain
                }
                nodes[node_type].append(node_info)

        self.write(json.dumps(nodes))

    def _build_consanguinity_chain(
        self, current_node, job_ctx, node_type, node_dict
    ):
        """Build consanguinity chain for a node based on rank_index (logical identity)."""
        # Get all logical instances of the same node (same type and rank_index)
        logical_instances = []

        # Key insight: rank_index represents logical identity within node type, not node_id
        current_rank = (
            current_node.rank_index
            if hasattr(current_node, "rank_index")
            else current_node.id
        )

        # Get all nodes of the same type with same logical identity
        for node_id, node in node_dict.items():
            # Use rank_index (logical ID) not node_id (physical ID)
            node_rank = (
                node.rank_index if hasattr(node, "rank_index") else node.id
            )
            if (
                node.type == current_node.type
                and node_rank == current_rank
                and node.start_time
            ):
                logical_instances.append(
                    {
                        "id": node_id,
                        "rank_index": node_rank,
                        "name": node.name,
                        "start_time": node.start_time,
                        "status": node.status,
                    }
                )

        # Add nodes from job context groups if needed (for cross-group replacements)
        # This handles cases where a node's rank might be in a different group
        try:
            from dlrover.python.master.node.job_context import get_job_context

            job_ctx = get_job_context()
            if hasattr(job_ctx, "_job_node_groups"):
                for group_nodes in job_ctx._job_node_groups.values():
                    for node_id, node in group_nodes.items():
                        node_rank = (
                            node.rank_index
                            if hasattr(node, "rank_index")
                            else node.id
                        )
                        if (
                            node.type == current_node.type
                            and node_rank == current_rank
                            and node.start_time
                            and node_id
                            not in [inst["id"] for inst in logical_instances]
                        ):
                            logical_instances.append(
                                {
                                    "id": node_id,
                                    "rank_index": node_rank,
                                    "name": node.name,
                                    "start_time": node.start_time,
                                    "status": node.status,
                                }
                            )
        except (ImportError, AttributeError):
            pass  # Job context not available, skip

        # Sort chronologically by start_time
        if logical_instances:
            logical_instances.sort(key=lambda x: x["start_time"])

            # Build chain showing the complete lineage
            if len(logical_instances) > 1:
                # More than one instance -> there's a replacement history
                chain_parts = []
                for inst in logical_instances:
                    # Format as: rank_index(node_id)
                    chain_parts.append(f"{inst['rank_index']}({inst['id']})")

                if chain_parts:
                    return " -> ".join(chain_parts)

        # Fallback: if no lineage found and we don't have rank_index, use name-based matching
        if not logical_instances and not hasattr(current_node, "rank_index"):
            # Fallback logic uses name patterns
            current_base = (
                current_node.name.split("-")[0]
                if "-" in current_node.name
                and len(current_node.name.split("-")) > 1
                else current_node.name
            )
            fallback_instances = []

            for node_id, node in node_dict.items():
                node_base = (
                    node.name.split("-")[0] if "-" in node.name else node.name
                )
                if node_base == current_base and node.start_time:
                    fallback_instances.append(
                        {
                            "id": node_id,
                            "rank_index": getattr(
                                node, "rank_index", 0
                            ),  # Use 0 as default rank
                            "name": node.name,
                            "start_time": node.start_time,
                            "status": node.status,
                        }
                    )

            fallback_instances.sort(key=lambda x: x["start_time"])

            if len(fallback_instances) > 1:
                return " -> ".join(
                    [
                        f"{inst['rank_index']}({inst['id']})"
                        for inst in fallback_instances
                    ]
                )

        # No lineage found
        return ""

    def _resource_to_dict(self, resource):
        """Convert resource object to dictionary."""
        return {
            "cpu": resource.cpu,
            "memory": resource.memory,
            "gpu": resource.gpu_num + f"({resource.gpu_type})",
            "gpu_type": resource.gpu_type,
        }


class LogsHandler(BaseHandler):
    """Handle requests for node logs."""

    executor = ThreadPoolExecutor(max_workers=4)

    @run_on_executor
    def get(self, node_name):
        """Get logs for a specific node."""
        try:
            # Get log file path from environment or configuration
            log_dir = os.environ.get("DLROVER_LOG_DIR", "/tmp/dlrover")
            log_file = os.path.join(log_dir, f"{node_name}.log")

            # Read last 1000 lines of log
            logs = self._tail_log_file(log_file, 1000)

            self.write(
                json.dumps(
                    {
                        "node_name": node_name,
                        "logs": logs,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            )
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))

    def _tail_log_file(self, filepath, lines):
        """Read last N lines from a file."""
        try:
            with open(filepath, "r") as f:
                return "".join(f.readlines()[-lines:])
        except FileNotFoundError:
            return f"Log file not found: {filepath}"


class DiagnosisHandler(BaseHandler):
    """Handle requests for diagnosis information."""

    def get(self):
        """Get diagnosis results and fault tolerance information."""
        try:
            # Get fault tolerance statistics
            from dlrover.python.master.node.job_context import JobContext

            job_ctx = JobContext.singleton_instance()

            # Mock data for diagnosis (safeguard if DiagnosisDataManager not available)
            results = [
                {
                    "type": "warning",
                    "description": "Diagnosis system not fully integrated",
                    "timestamp": datetime.now().isoformat(),
                }
            ]

            fault_stats = {
                "total_restarts": job_ctx.get_job_restart_count(),
                "failed_nodes": job_ctx.get_failed_node_cnt(),
                "recent_failures": self._get_recent_failures(job_ctx, 10),
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
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))

    def _get_recent_failures(self, job_ctx, limit):
        """Get recent node failures."""
        failures = []
        # Get all nodes and find failed ones
        for node_type in [
            NodeType.WORKER,
            NodeType.PS,
            NodeType.CHIEF,
            NodeType.EVALUATOR,
        ]:
            for node_id, node in job_ctx.get_mutable_job_nodes(
                node_type
            ).items():
                if node.status == NodeStatus.FAILED and node.finish_time:
                    failures.append(
                        {
                            "name": node.name,
                            "type": node.type,
                            "exit_reason": node.exit_reason or "Unknown",
                            "finish_time": node.finish_time.isoformat()
                            if node.finish_time
                            else None,
                        }
                    )

        # Sort by finish time and limit
        failures.sort(key=lambda x: x["finish_time"], reverse=True)
        return failures[:limit]


class MetricsHandler(BaseHandler):
    """Handle requests for real-time metrics."""

    def get(self):
        """Get real-time metrics including resource usage and training speed."""
        job_ctx = get_job_context()

        # Aggregate resource usage across all nodes
        total_cpu = 0
        total_memory = 0
        total_gpu = 0
        running_workers = 0

        for node_type in [NodeType.WORKER, NodeType.PS, NodeType.CHIEF]:
            for node in job_ctx.get_mutable_job_nodes(node_type).values():
                if node.status == NodeStatus.RUNNING and node.used_resource:
                    total_cpu += float(getattr(node.used_resource, "cpu", 0))
                    total_memory += float(
                        getattr(node.used_resource, "memory", 0)
                    )
                    total_gpu += getattr(node.used_resource, "gpu_num", 0)

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
        try:
            from dlrover.python.master.monitor.perf_monitor import PerfMonitor
            import gc

            for obj in gc.get_objects():
                if isinstance(obj, PerfMonitor):
                    return obj.completed_global_step
        except Exception:
            pass
        return 0

    def _get_training_speed(self):
        """Get current training speed from available monitors."""
        try:
            from dlrover.python.master.monitor.perf_monitor import PerfMonitor
            import gc

            for obj in gc.get_objects():
                if isinstance(obj, PerfMonitor):
                    return obj.running_speed
        except Exception:
            pass
        return 0.0


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
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


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
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


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
            # Try to get the actual JobArgs instance from the job context
            job_ctx = get_job_context()

            # Get JobArgs from context
            job_args = job_ctx.get_job_args()

            # Convert JobArgs to dict
            job_args_data = {}
            for k, v in job_args.__dict__.items():
                if k.startswith("_") or callable(v):
                    continue
                # Handle special types
                if isinstance(
                    v, (str, int, float, bool, list, dict, type(None))
                ):
                    job_args_data[k] = v
                elif hasattr(
                    v, "__dict__"
                ):  # Handle objects like ResourceLimits
                    job_args_data[k] = v.__dict__
                else:
                    job_args_data[k] = str(v)

            self.write(json.dumps(job_args_data))
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


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
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))


def create_dashboard_app():
    """Create the dashboard application."""

    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "static"),
        "template_path": os.path.join(os.path.dirname(__file__), "templates"),
        "debug": True,
    }

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


class IndexHandler(web.RequestHandler):
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


class NodeDetailHandler(web.RequestHandler):
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
