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

from dlrover.dashboard.integrate_with_master import (
    DashboardManager,
    add_dashboard_to_master,
)


class TestDashboardManagerInit(unittest.TestCase):
    """Test DashboardManager initialization."""

    def test_default_init(self):
        mgr = DashboardManager()
        self.assertEqual(mgr.host, "0.0.0.0")
        self.assertEqual(mgr.port, 8080)
        self.assertTrue(mgr.enable)
        self.assertIsNone(mgr._perf_monitor)
        self.assertIsNone(mgr._dashboard_thread)
        self.assertFalse(mgr._stop_event.is_set())

    def test_custom_init(self):
        monitor = MagicMock()
        mgr = DashboardManager(
            host="127.0.0.1", port=9090, enable=False, perf_monitor=monitor
        )
        self.assertEqual(mgr.host, "127.0.0.1")
        self.assertEqual(mgr.port, 9090)
        self.assertFalse(mgr.enable)
        self.assertIs(mgr._perf_monitor, monitor)


class TestDashboardManagerStart(unittest.TestCase):
    """Test DashboardManager.start()."""

    @patch("dlrover.dashboard.integrate_with_master.logger")
    def test_start_disabled(self, mock_logger):
        mgr = DashboardManager(enable=False)
        mgr.start()
        mock_logger.info.assert_any_call("Dashboard is disabled")
        self.assertIsNone(mgr._dashboard_thread)

    @patch.object(DashboardManager, "_run_dashboard_server")
    @patch.object(DashboardManager, "_broadcast_loop")
    def test_start_enabled(self, mock_broadcast, mock_server):
        mgr = DashboardManager(enable=True)
        mgr.start()
        self.assertIsNotNone(mgr._dashboard_thread)
        self.assertTrue(mgr._dashboard_thread.daemon)
        # Wait briefly for thread to invoke target
        mgr._dashboard_thread.join(timeout=1)
        mgr.stop()

    @patch("dlrover.dashboard.integrate_with_master.logger")
    @patch("dlrover.dashboard.integrate_with_master.threading.Thread")
    def test_start_exception(self, mock_thread_cls, mock_logger):
        mock_thread_cls.side_effect = RuntimeError("thread error")
        mgr = DashboardManager(enable=True)
        # Should not raise
        mgr.start()
        mock_logger.error.assert_called()


class TestDashboardManagerStop(unittest.TestCase):
    """Test DashboardManager.stop()."""

    def test_stop_sets_event(self):
        mgr = DashboardManager()
        self.assertFalse(mgr._stop_event.is_set())
        mgr.stop()
        self.assertTrue(mgr._stop_event.is_set())

    @patch("dlrover.dashboard.integrate_with_master.logger")
    def test_stop_exception(self, mock_logger):
        mgr = DashboardManager()
        mgr._executor = MagicMock()
        mgr._executor.shutdown.side_effect = RuntimeError("shutdown err")
        mgr.stop()
        mock_logger.error.assert_called()


class TestDashboardManagerRunServer(unittest.TestCase):
    """Test DashboardManager._run_dashboard_server()."""

    @patch("dlrover.dashboard.integrate_with_master.create_dashboard_app")
    def test_run_dashboard_server(self, mock_create_app):
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        with patch(
            "dlrover.dashboard.integrate_with_master.DashboardManager"
            "._run_dashboard_server"
        ):
            # Test the real method by calling it directly with mocked imports
            pass

        # Test via patching tornado internals
        mgr = DashboardManager(host="127.0.0.1", port=9999)
        with patch("tornado.httpserver.HTTPServer") as mock_server_cls:
            with patch("tornado.ioloop.IOLoop") as mock_ioloop:
                mock_server = MagicMock()
                mock_server_cls.return_value = mock_server
                mock_loop = MagicMock()
                mock_ioloop.current.return_value = mock_loop

                mgr._run_dashboard_server()

                mock_create_app.assert_called_once_with(perf_monitor=None)
                mock_server.listen.assert_called_once_with(9999, "127.0.0.1")
                mock_loop.start.assert_called_once()

    @patch("dlrover.dashboard.integrate_with_master.create_dashboard_app")
    @patch("dlrover.dashboard.integrate_with_master.logger")
    def test_run_dashboard_server_exception(self, mock_logger, mock_create):
        mock_create.side_effect = RuntimeError("app error")
        mgr = DashboardManager()
        mgr._run_dashboard_server()
        mock_logger.error.assert_called()


class TestDashboardManagerBroadcastLoop(unittest.TestCase):
    """Test DashboardManager._broadcast_loop()."""

    @patch("tornado.ioloop.IOLoop")
    @patch("dlrover.dashboard.integrate_with_master.WebSocketHandler")
    @patch("dlrover.dashboard.integrate_with_master.JobContext")
    def test_broadcast_loop_single_iteration(
        self, mock_jc_cls, mock_ws, mock_ioloop_cls
    ):
        mock_ctx = MagicMock()
        mock_ctx.get_job_stage.return_value = "running"
        mock_ctx.get_running_node_size.return_value = 4
        mock_ctx.get_failed_node_size.return_value = 0
        mock_ctx.get_total_node_size.return_value = 4
        mock_jc_cls.singleton_instance.return_value = mock_ctx

        mock_loop = MagicMock()
        mock_ioloop_cls.current.return_value = mock_loop

        mgr = DashboardManager()

        call_count = 0

        def stop_after_first(timeout):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                mgr._stop_event.set()
            return True

        mgr._stop_event.wait = stop_after_first
        mgr._broadcast_loop()

        mock_loop.add_callback.assert_called_once()
        args = mock_loop.add_callback.call_args[0]
        self.assertIs(args[0], mock_ws.broadcast)
        self.assertEqual(call_count, 1)

    @patch("dlrover.dashboard.integrate_with_master.WebSocketHandler")
    @patch("dlrover.dashboard.integrate_with_master.JobContext")
    @patch("dlrover.dashboard.integrate_with_master.logger")
    def test_broadcast_loop_exception_backoff(
        self, mock_logger, mock_jc_cls, mock_ws
    ):
        mock_jc_cls.singleton_instance.side_effect = RuntimeError("ctx err")

        mgr = DashboardManager()
        call_count = 0

        def stop_after_two(timeout):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                mgr._stop_event.set()
            return True

        mgr._stop_event.wait = stop_after_two
        mgr._broadcast_loop()

        # Should have logged errors
        self.assertTrue(mock_logger.error.called)

    @patch("dlrover.dashboard.integrate_with_master.WebSocketHandler")
    @patch("dlrover.dashboard.integrate_with_master.JobContext")
    @patch("dlrover.dashboard.integrate_with_master.logger")
    def test_broadcast_loop_max_failures_breaks(
        self, mock_logger, mock_jc_cls, mock_ws
    ):
        mock_jc_cls.singleton_instance.side_effect = RuntimeError("err")

        mgr = DashboardManager()
        # Never set stop event; rely on max_consecutive_failures to break
        mgr._stop_event.wait = lambda timeout: False

        mgr._broadcast_loop()

        # Should log the "too many consecutive" message
        error_calls = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("Too many consecutive" in c for c in error_calls))


class TestAddDashboardToMaster(unittest.TestCase):
    """Test add_dashboard_to_master function."""

    def test_none_master_returns_none(self):
        result = add_dashboard_to_master(None, {"enable": True})
        self.assertIsNone(result)

    def test_none_config_uses_defaults(self):
        master = MagicMock()
        result = add_dashboard_to_master(master, None)
        self.assertIsNotNone(result)
        self.assertEqual(result.port, 8080)
        self.assertEqual(result.host, "0.0.0.0")
        self.assertTrue(result.enable)

    def test_empty_config_uses_defaults(self):
        master = MagicMock()
        result = add_dashboard_to_master(master, {})
        self.assertIsNotNone(result)
        self.assertEqual(result.port, 8080)

    def test_disabled_returns_none(self):
        master = MagicMock()
        result = add_dashboard_to_master(
            master, {"enable": False, "port": 9090}
        )
        self.assertIsNone(result)

    def test_returns_dashboard_manager(self):
        master = MagicMock()
        result = add_dashboard_to_master(
            master, {"enable": True, "host": "127.0.0.1", "port": 9090}
        )
        self.assertIsInstance(result, DashboardManager)
        self.assertEqual(result.host, "127.0.0.1")
        self.assertEqual(result.port, 9090)

    def test_stores_instance_on_master(self):
        master = MagicMock()
        result = add_dashboard_to_master(
            master, {"enable": True, "port": 8080}
        )
        self.assertIs(master._dashboard_instance, result)

    def test_perf_monitor_forwarded(self):
        master = MagicMock()
        monitor = MagicMock()
        master.perf_monitor = monitor
        result = add_dashboard_to_master(
            master, {"enable": True, "port": 8080}
        )
        self.assertIs(result._perf_monitor, monitor)

    def test_perf_monitor_missing(self):
        master = MagicMock()
        # Remove perf_monitor so getattr returns None
        del master.perf_monitor
        result = add_dashboard_to_master(
            master, {"enable": True, "port": 8080}
        )
        self.assertIsNone(result._perf_monitor)


class TestLifecycleHooks(unittest.TestCase):
    """Test prepare_with_dashboard and stop_with_dashboard hooks."""

    def test_prepare_hook_starts_dashboard_then_calls_original(self):
        master = MagicMock()
        original_prepare = master.prepare
        add_dashboard_to_master(master, {"enable": True, "port": 8080})

        call_order = []
        master._dashboard_instance.start = lambda: call_order.append("dash")
        original_prepare.side_effect = lambda: call_order.append("prepare")

        master.prepare()

        self.assertEqual(call_order, ["dash", "prepare"])

    def test_stop_hook_stops_original_then_dashboard(self):
        master = MagicMock()
        original_stop = master.stop
        add_dashboard_to_master(master, {"enable": True, "port": 8080})

        call_order = []
        original_stop.side_effect = lambda: call_order.append("stop")
        master._dashboard_instance.stop = lambda: call_order.append("dash")

        master.stop()

        self.assertEqual(call_order, ["stop", "dash"])

    def test_prepare_hook_without_dashboard_instance(self):
        master = MagicMock()
        original_prepare = MagicMock()
        master.prepare = original_prepare
        add_dashboard_to_master(master, {"enable": True, "port": 8080})

        # Remove dashboard instance to test None guard
        master._dashboard_instance = None
        master.prepare()
        # original_prepare should still be called (via closure)

    def test_stop_hook_without_dashboard_instance(self):
        master = MagicMock()
        original_stop = MagicMock()
        master.stop = original_stop
        add_dashboard_to_master(master, {"enable": True, "port": 8080})

        master._dashboard_instance = None
        master.stop()


if __name__ == "__main__":
    unittest.main()
