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
from collections import deque

import requests

from dlrover.python.common import env_utils
from dlrover.python.common.log import default_logger as logger
from dlrover.python.diagnosis.common.constants import (
    DiagnosisConstant,
    DiagnosisActionType,
    EnvConfigKey,
)
from dlrover.python.diagnosis.common.diagnosis_action import NodeAction


DEFAULT_PROFILING_DURATION_SECONDS = (
    DiagnosisConstant.PROFILING_DEFAULT_DURATION_SECS
)
DEFAULT_MAX_PROFILING_DURATION_SECONDS = (
    DiagnosisConstant.PROFILING_MAX_DURATION_SECS
)
DEFAULT_XPU_TIMER_CONTROL_PORT = 18888
DEFAULT_REQUEST_TIMEOUT_SECONDS = 2
DEFAULT_REQUEST_RETRIES = 2
DEFAULT_STOP_RETRY_INTERVAL_SECONDS = 1
MAX_EXECUTED_ACTION_IDS = 1024


class ProfilerController:
    """Control XPU Timer for one worker node."""

    def __init__(
        self,
        node_id=None,
        control_url=None,
        default_duration_seconds=DEFAULT_PROFILING_DURATION_SECONDS,
        max_duration_seconds=DEFAULT_MAX_PROFILING_DURATION_SECONDS,
        request_timeout_seconds=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        request_retries=DEFAULT_REQUEST_RETRIES,
        stop_retry_interval_seconds=DEFAULT_STOP_RETRY_INTERVAL_SECONDS,
        post_func=None,
    ):
        self._node_id = env_utils.get_node_id() if node_id is None else node_id
        self._control_url = control_url or self._get_control_url()
        self._default_duration_seconds = max(1, default_duration_seconds)
        self._max_duration_seconds = max(
            self._default_duration_seconds, max_duration_seconds
        )
        self._request_timeout_seconds = max(0.1, request_timeout_seconds)
        self._request_retries = max(1, request_retries)
        self._stop_retry_interval_seconds = max(0, stop_retry_interval_seconds)
        self._post_func = post_func or requests.post
        self._lock = threading.RLock()
        self._active_action_id = ""
        self._active_action_group_id = ""
        self._deadline = 0.0
        self._session_stop_event = None
        self._watchdog = None
        self._executed_action_ids = set()
        self._executed_order = deque(maxlen=MAX_EXECUTED_ACTION_IDS)

    @property
    def is_active(self):
        with self._lock:
            return bool(self._active_action_id)

    @property
    def active_action_id(self):
        with self._lock:
            return self._active_action_id

    def enable(
        self,
        action: NodeAction,
        current_run_id="",
        current_rdzv_epoch=None,
    ):
        reason = self._validation_error(
            action, current_run_id, current_rdzv_epoch
        )
        if reason:
            logger.warning(
                f"Reject profiling action "
                f"{getattr(action, 'action_id', '')}: {reason}."
            )
            return False

        with self._lock:
            if action.action_id in self._executed_action_ids:
                logger.info(
                    f"Profiling action {action.action_id} was already handled."
                )
                return True
            if self._active_action_id:
                logger.warning(
                    f"Profiling action {action.action_id} conflicts with "
                    f"active action {self._active_action_id}."
                )
                return False

            duration = self._duration_seconds(action)
            self._active_action_id = action.action_id
            self._active_action_group_id = action.action_group_id

        try:
            self._set_enabled(True)
        except Exception as error:
            logger.warning(
                f"Failed to enable profiling with action "
                f"{action.action_id}: {error}"
            )
            try:
                self._set_enabled(False, retries=1)
            except Exception as disable_error:
                logger.warning(
                    f"Failed to clean up profiling state after start "
                    f"failure: {disable_error}"
                )
            self._clear_active_session()
            return False

        with self._lock:
            self._remember_action(action.action_id)
            stop_event = threading.Event()
            self._session_stop_event = stop_event
            self._deadline = time.monotonic() + duration
            self._watchdog = threading.Thread(
                target=self._watch_session,
                args=(action.action_id, stop_event, duration),
                name=f"profiler-stop-{action.action_id}",
                daemon=True,
            )
            self._watchdog.start()

        logger.info(
            f"Enabled profiling for worker {self._node_id} with action "
            f"{action.action_id} for {duration} seconds."
        )
        return True

    def disable(self, reason="requested"):
        with self._lock:
            action_id = self._active_action_id
            stop_event = self._session_stop_event
        if not action_id:
            return True
        return self._stop_session(action_id, stop_event, reason)

    def shutdown(self):
        with self._lock:
            stop_event = self._session_stop_event
        if stop_event is not None:
            stop_event.set()
        self.disable(reason="agent shutdown")

    def _watch_session(self, action_id, stop_event, duration):
        if stop_event.wait(duration):
            return
        self._stop_session(action_id, stop_event, "duration elapsed")

    def _stop_session(self, action_id, stop_event, reason):
        if stop_event is not None:
            stop_event.set()
        with self._lock:
            if self._active_action_id != action_id:
                return True

        try:
            self._set_enabled(False)
        except Exception as error:
            logger.warning(
                f"Failed to disable profiling for action {action_id}: {error}"
            )
            self._clear_active_session(action_id)
            return False

        self._clear_active_session(action_id)
        logger.info(
            f"Disabled profiling for worker {self._node_id} with action "
            f"{action_id}: {reason}."
        )
        return True

    def _set_enabled(self, enabled, retries=None):
        last_error = None
        attempt_count = (
            self._request_retries if retries is None else max(1, retries)
        )
        for attempt in range(attempt_count):
            try:
                response = self._post_func(
                    self._control_url,
                    json={"enabled": enabled},
                    timeout=self._request_timeout_seconds,
                )
                if response is None:
                    raise RuntimeError("XPU Timer returned no response")
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
                elif hasattr(response, "ok") and not response.ok:
                    raise RuntimeError(
                        f"XPU Timer returned HTTP {response.status_code}"
                    )
                return
            except Exception as error:
                last_error = error
                if attempt + 1 < attempt_count:
                    time.sleep(self._stop_retry_interval_seconds)
        raise RuntimeError(str(last_error))

    def _clear_active_session(self, action_id=None):
        with self._lock:
            if action_id is not None and self._active_action_id != action_id:
                return
            stop_event = self._session_stop_event
            if stop_event is not None:
                stop_event.set()
            self._active_action_id = ""
            self._active_action_group_id = ""
            self._deadline = 0.0
            self._session_stop_event = None
            self._watchdog = None

    def _remember_action(self, action_id):
        if action_id in self._executed_action_ids:
            return
        if len(self._executed_order) == self._executed_order.maxlen:
            expired_action_id = self._executed_order.popleft()
            self._executed_action_ids.discard(expired_action_id)
        self._executed_order.append(action_id)
        self._executed_action_ids.add(action_id)

    def _duration_seconds(self, action):
        value = action.parameters.get(
            "duration_seconds", self._default_duration_seconds
        )
        try:
            duration = int(value)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid profiling duration {value}; use "
                f"{self._default_duration_seconds} seconds."
            )
            duration = self._default_duration_seconds
        duration = max(1, duration)
        if duration > self._max_duration_seconds:
            logger.warning(
                f"Clamp profiling duration {duration} to "
                f"{self._max_duration_seconds} seconds."
            )
            duration = self._max_duration_seconds
        return duration

    def _validation_error(self, action, current_run_id, current_rdzv_epoch):
        if not isinstance(action, NodeAction):
            return "action is not a NodeAction"
        if action.action_type != DiagnosisActionType.ENABLE_PROFILING:
            return "unsupported action type"
        if not action.action_id:
            return "action has no action ID"
        if action.is_expired():
            return "action is expired"
        if action.node_id != self._node_id:
            return "action targets another node"
        if action.instance not in (self._node_id, -3):
            return "action is routed to another instance"

        job_uid = env_utils.get_job_uid()
        if job_uid and action.job_uid and action.job_uid != job_uid:
            return "action belongs to another job"
        if (
            current_run_id
            and action.run_id
            and action.run_id != current_run_id
        ):
            return "action belongs to another run"
        if (
            current_rdzv_epoch is not None
            and action.rdzv_epoch != current_rdzv_epoch
        ):
            return "action belongs to another rendezvous epoch"
        return ""

    @staticmethod
    def _get_control_url():
        configured_url = env_utils.get_env(EnvConfigKey.XPU_TIMER_CONTROL_URL)
        if configured_url:
            return configured_url

        control_port = env_utils.get_env(EnvConfigKey.XPU_TIMER_CONTROL_PORT)
        if control_port:
            port = (
                ProfilerController._parse_port(control_port)
                or DEFAULT_XPU_TIMER_CONTROL_PORT
            )
        else:
            metric_port = env_utils.get_env(EnvConfigKey.XPU_TIMER_PORT)
            port = DEFAULT_XPU_TIMER_CONTROL_PORT
            if metric_port:
                metric_port_value = ProfilerController._parse_port(metric_port)
                if metric_port_value > 1:
                    port = metric_port_value - 1
        return f"http://127.0.0.1:{port}/HostingService/UpdateCuptiState"

    @staticmethod
    def _parse_port(value):
        try:
            port = int(value)
        except (TypeError, ValueError):
            port = 0
        if port < 1 or port > 65535:
            logger.warning(
                f"Invalid XPU Timer control port {value}; use "
                f"{DEFAULT_XPU_TIMER_CONTROL_PORT}."
            )
            return 0
        return port
