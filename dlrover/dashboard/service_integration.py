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

import threading
import time
from datetime import datetime
from typing import Optional

from dlrover.python.common.log import default_logger as logger
from dlrover.python.master.node.job_context import JobContext


class DashboardService:
    """Service to integrate dashboard with dlrover master."""

    def __init__(self, perf_monitor=None):
        self._stop = False
        self._monitor_thread = None
        self._global_step = 0
        self._last_step_time = time.time()
        self._training_speed = 0.0
        self._total_steps = 0
        self._perf_monitor = perf_monitor

    def start(self):
        """Start the dashboard service."""
        if self._monitor_thread is None:
            self._stop = False
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("Dashboard service started")

    def stop(self):
        """Stop the dashboard service."""
        self._stop = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            logger.info("Dashboard service stopped")

    def _monitor_loop(self):
        """Main monitoring loop to update training metrics."""
        while not self._stop:
            try:
                # Update training metrics
                self._update_training_metrics()

                # Sleep for 1 second
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in dashboard monitor loop: {e}")
                time.sleep(5)  # Wait longer on error

    def _update_training_metrics(self):
        """Update training metrics from PerfMonitor."""

        # Update from internal PerfMonitor instance if available
        if self._perf_monitor:
            self._global_step = self._perf_monitor.completed_global_step
            self._training_speed = self._perf_monitor.running_speed
            self._total_steps = self._perf_monitor.completed_global_step
        else:
            # Fallback: try to find PerfMonitor in global objects
            try:
                from dlrover.python.master.monitor.perf_monitor import (
                    PerfMonitor,
                )

                monitor = None
                import gc

                for obj in gc.get_objects():
                    if isinstance(obj, PerfMonitor):
                        monitor = obj
                        break

                if monitor:
                    self._global_step = monitor.completed_global_step
                    self._training_speed = monitor.running_speed
                    self._total_steps = monitor.completed_global_step
            except Exception as e:
                logger.debug(f"PerfMonitor not available yet: {e}")

    def _update_step_metrics(self, current_step):
        """Update step-based metrics."""
        if current_step > self._global_step:
            # Calculate training speed
            current_time = time.time()
            time_diff = current_time - self._last_step_time
            step_diff = current_step - self._global_step

            if time_diff > 0:
                self._training_speed = step_diff / time_diff

            # Update state
            self._global_step = current_step
            self._last_step_time = current_time
            self._total_steps = current_step

    def get_global_step(self) -> int:
        """Get current global step."""
        return self._global_step

    def get_training_speed(self) -> float:
        """Get current training speed in steps/second."""
        return self._training_speed

    def get_total_steps(self) -> int:
        """Get total steps completed."""
        return self._total_steps

    def get_session_info(self) -> dict:
        """Get session information for dashboard."""
        job_ctx = JobContext.singleton_instance()

        return {
            "job_name": job_ctx.get_job_name(),
            "job_type": job_ctx.get_job_type(),
            "job_stage": job_ctx.get_job_stage(),
            "create_time": job_ctx.get_job_create_time(),
            "start_time": job_ctx.get_job_start_time(),
            "global_step": self.get_global_step(),
            "training_speed": self.get_training_speed(),
            "total_steps": self.get_total_steps(),
            "timestamp": datetime.now(),
        }


# Global dashboard service instance
_dashboard_service: Optional[DashboardService] = None


def get_dashboard_service(perf_monitor=None) -> DashboardService:
    """Get the global dashboard service instance."""
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService(perf_monitor)
    return _dashboard_service


def start_dashboard_service(perf_monitor=None):
    """Start the dashboard service."""
    service = get_dashboard_service(perf_monitor)
    service.start()
    return service


def stop_dashboard_service():
    """Stop the dashboard service."""
    global _dashboard_service
    if _dashboard_service:
        _dashboard_service.stop()
        _dashboard_service = None
