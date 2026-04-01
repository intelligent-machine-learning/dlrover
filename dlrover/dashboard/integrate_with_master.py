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
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.node.job_context import JobContext
from dlrover.dashboard.app import create_dashboard_app, WebSocketHandler


class DashboardManager:
    """Manager for dashboard integration in DLRover Master."""

    def __init__(
        self, host="0.0.0.0", port=8080, enable=True, perf_monitor=None
    ):
        self.host = host
        self.port = port
        self.enable = enable
        self._perf_monitor = perf_monitor
        self._dashboard_thread = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._stop_event = threading.Event()

    def start(self):
        """Start the dashboard manager."""
        if not self.enable:
            logger.info("Dashboard is disabled")
            return

        try:
            self._stop_event.clear()

            # Start dashboard server in a separate thread
            self._dashboard_thread = threading.Thread(
                target=self._run_dashboard_server
            )
            self._dashboard_thread.daemon = True
            self._dashboard_thread.start()

            # Start broadcast thread for real-time updates
            self._executor.submit(self._broadcast_loop)

            logger.info(
                f"Dashboard manager started on {self.host}:{self.port}"
            )
        except Exception as e:
            logger.error(f"Failed to start dashboard manager: {e}")

    def stop(self):
        """Stop the dashboard manager."""
        try:
            self._stop_event.set()
            self._executor.shutdown(wait=False)
            logger.info("Dashboard manager stopped")
        except Exception as e:
            logger.error(f"Error stopping dashboard manager: {e}")

    def _run_dashboard_server(self):
        """Run the dashboard server."""
        try:
            from tornado.httpserver import HTTPServer
            from tornado.ioloop import IOLoop

            app = create_dashboard_app(perf_monitor=self._perf_monitor)
            server = HTTPServer(app)
            server.listen(self.port, self.host)

            logger.info(
                f"Dashboard server listening on http://{self.host}:{self.port}"
            )
            IOLoop.current().start()
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")

    def _broadcast_loop(self):
        """Broadcast real-time updates to dashboard clients."""
        from tornado.ioloop import IOLoop

        consecutive_failures = 0
        max_consecutive_failures = 30

        while not self._stop_event.is_set():
            try:
                job_ctx = JobContext.singleton_instance()

                # Create update message
                update = {
                    "type": "status_update",
                    "data": {
                        "timestamp": time.time(),
                        "job_stage": job_ctx.get_job_stage(),
                        "running_nodes": job_ctx.get_running_node_size(),
                        "failed_nodes": job_ctx.get_failed_node_size(),
                        "total_nodes": job_ctx.get_total_node_size(),
                    },
                }

                # Schedule broadcast on the Tornado IOLoop thread to
                # avoid cross-thread WebSocket writes.
                message = json.dumps(update)
                IOLoop.current().add_callback(
                    WebSocketHandler.broadcast, message
                )

                consecutive_failures = 0
                self._stop_event.wait(2)
            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    f"Error in broadcast loop "
                    f"({consecutive_failures}/{max_consecutive_failures}): {e}"
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "Too many consecutive broadcast failures, "
                        "stopping broadcast loop"
                    )
                    break
                # Exponential backoff capped at 30 seconds
                backoff = min(5 * (2 ** (consecutive_failures - 1)), 30)
                self._stop_event.wait(backoff)


def add_dashboard_to_master(master_instance, dashboard_config=None):
    """Add dashboard support to DLRover Master server.

    Args:
        master_instance: The DLRover Master server instance
        dashboard_config: Optional configuration for dashboard
    """
    if not master_instance:
        return None
    if not dashboard_config:
        dashboard_config = {"enable": True, "host": "0.0.0.0", "port": 8080}
    if not dashboard_config.get("enable", True):
        return None

    # Create dashboard instance
    instance = DashboardManager(
        host=dashboard_config.get("host", "0.0.0.0"),
        port=dashboard_config.get("port", 8080),
        enable=dashboard_config.get("enable", True),
        perf_monitor=getattr(master_instance, "perf_monitor", None),
    )

    # Store on master server
    master_instance._dashboard_instance = instance

    # Add hooks to master lifecycle
    original_prepare = master_instance.prepare
    original_stop = master_instance.stop

    def prepare_with_dashboard():
        """Start master with dashboard."""
        logger.info("Starting DLRover Master with Dashboard support")

        # Start dashboard first
        if master_instance._dashboard_instance:
            master_instance._dashboard_instance.start()

        # Start original master
        original_prepare()

    def stop_with_dashboard():
        """Stop master with dashboard."""
        logger.info("Stopping DLRover Master with Dashboard")

        # Stop original master first
        original_stop()

        # Stop dashboard
        if master_instance._dashboard_instance:
            master_instance._dashboard_instance.stop()

    # Replace methods with consistent naming
    master_instance.prepare = prepare_with_dashboard
    master_instance.stop = stop_with_dashboard

    logger.info(
        f"Dashboard integration added to master on port {instance.port}"
    )
    return instance
