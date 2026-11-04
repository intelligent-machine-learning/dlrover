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

import os
import time
import types
import unittest
from unittest.mock import MagicMock, patch

from dlrover.python.diagnosis.common.constants import DiagnosisActionType
from dlrover.python.diagnosis.common.diagnosis_action import NodeAction
from dlrover.python.elastic_agent.profiler_controller import (
    ProfilerController,
)


class ProfilerControllerTest(unittest.TestCase):
    def setUp(self):
        env_patcher = patch.dict(os.environ, {"JOB_UID": "job-1"})
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        self.response = MagicMock()
        self.post = MagicMock(return_value=self.response)

    def _controller(self):
        return ProfilerController(
            node_id=0,
            control_url=(
                "http://127.0.0.1:18888/HostingService/UpdateCuptiState"
            ),
            request_retries=1,
            stop_retry_interval_seconds=0,
            post_func=self.post,
        )

    def _action(
        self,
        action_id="action-1",
        node_id=0,
        duration="10",
        epoch=2,
    ):
        return NodeAction(
            node_id=node_id,
            node_type="worker",
            instance=node_id,
            action_type=DiagnosisActionType.ENABLE_PROFILING,
            action_id=action_id,
            action_group_id="performance-1",
            diagnosis_id="performance-1",
            job_uid="job-1",
            run_id="run-1",
            rdzv_epoch=epoch,
            parameters={"duration_seconds": duration},
        )

    def test_enable_duplicate_and_explicit_disable(self):
        controller = self._controller()
        action = self._action()
        self.assertTrue(
            controller.enable(
                action,
                current_run_id="run-1",
                current_rdzv_epoch=2,
            )
        )
        self.assertTrue(
            controller.enable(
                action,
                current_run_id="run-1",
                current_rdzv_epoch=2,
            )
        )
        self.assertEqual(
            self.post.call_args_list[0].kwargs["json"],
            {"enabled": True},
        )
        self.assertEqual(
            self.post.call_args_list[0].args[0],
            "http://127.0.0.1:18888/HostingService/UpdateCuptiState",
        )
        self.assertEqual(self.post.call_count, 1)

        self.assertTrue(controller.disable())
        self.assertFalse(controller.is_active)
        self.assertEqual(
            self.post.call_args_list[-1].kwargs["json"],
            {"enabled": False},
        )
        self.assertEqual(self.post.call_count, 2)

    def test_duration_watchdog_disables_profiling(self):
        controller = self._controller()
        self.assertTrue(
            controller.enable(
                self._action(duration="1"),
                current_run_id="run-1",
                current_rdzv_epoch=2,
            )
        )

        deadline = time.time() + 3
        while self.post.call_count < 2 and time.time() < deadline:
            time.sleep(0.05)

        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(
            self.post.call_args_list[-1].kwargs["json"],
            {"enabled": False},
        )
        self.assertFalse(controller.is_active)

    def test_reject_stale_or_wrong_target_action(self):
        controller = self._controller()
        stale_action = self._action()
        stale_action._timestamp -= stale_action.expired_time_period + 1
        self.assertFalse(controller.enable(stale_action))
        self.assertFalse(controller.enable(self._action(node_id=1)))
        self.assertFalse(
            controller.enable(
                self._action(),
                current_rdzv_epoch=3,
            )
        )
        self.post.assert_not_called()

    def test_start_failure_leaves_controller_idle(self):
        self.post.side_effect = RuntimeError("connection refused")
        controller = self._controller()
        self.assertFalse(controller.enable(self._action()))
        self.assertFalse(controller.is_active)
        self.assertEqual(self.post.call_count, 2)

    def test_control_url_and_duration_validation(self):
        with patch.dict(
            os.environ,
            {
                "XPU_TIMER_CONTROL_URL": "",
                "XPU_TIMER_CONTROL_PORT": "",
                "XPU_TIMER_PORT": "18889",
            },
        ):
            controller = ProfilerController(post_func=self.post)
        self.assertEqual(
            controller._control_url,
            "http://127.0.0.1:18888/HostingService/UpdateCuptiState",
        )
        self.assertEqual(controller._parse_port("invalid"), 0)
        self.assertEqual(
            controller._duration_seconds(self._action(duration="invalid")),
            180,
        )
        self.assertEqual(
            controller._duration_seconds(self._action(duration="10000")),
            600,
        )

    def test_shutdown_retries_disable_and_clears_session(self):
        controller = ProfilerController(
            node_id=0,
            control_url="http://127.0.0.1:18888/test",
            request_retries=2,
            stop_retry_interval_seconds=0,
            post_func=self.post,
        )
        self.post.side_effect = [
            self.response,
            RuntimeError("disable failed"),
            RuntimeError("disable failed again"),
        ]
        self.assertTrue(controller.enable(self._action()))
        controller.shutdown()

        self.assertFalse(controller.is_active)
        self.assertTrue(controller.disable())
        self.assertEqual(self.post.call_count, 3)

    def test_empty_response_is_a_start_failure(self):
        self.post.return_value = None
        controller = self._controller()
        self.assertFalse(controller.enable(self._action()))
        self.assertFalse(controller.is_active)
        self.assertEqual(self.post.call_count, 2)

    def test_validation_rejections(self):
        controller = self._controller()
        self.assertFalse(
            controller.enable(None)  # type: ignore[arg-type]
        )

        wrong_type = self._action()
        wrong_type._action_type = DiagnosisActionType.COLLECT_METRIC
        self.assertFalse(controller.enable(wrong_type))

        missing_id = self._action()
        missing_id._action_id = ""
        self.assertFalse(controller.enable(missing_id))

        wrong_instance = self._action()
        wrong_instance._instance = 9
        self.assertFalse(controller.enable(wrong_instance))

        self.assertFalse(
            controller.enable(self._action(), current_run_id="other-run")
        )
        wrong_job = self._action()
        wrong_job._job_uid = "other-job"
        self.assertFalse(controller.enable(wrong_job))
        self.post.assert_not_called()

    def test_configured_url_response_status_and_action_cache(self):
        with patch.dict(
            os.environ,
            {"XPU_TIMER_CONTROL_URL": "http://127.0.0.1:9999/test"},
        ):
            controller = ProfilerController(
                node_id=0,
                request_retries=1,
                post_func=self.post,
            )
        self.assertEqual(controller._control_url, "http://127.0.0.1:9999/test")

        self.post.return_value = types.SimpleNamespace(
            ok=False,
            status_code=500,
        )
        self.assertFalse(controller.enable(self._action()))

        for index in range(1025):
            controller._remember_action(f"action-{index}")
        self.assertEqual(len(controller._executed_action_ids), 1024)
        self.assertNotIn("action-0", controller._executed_action_ids)


if __name__ == "__main__":
    unittest.main()
