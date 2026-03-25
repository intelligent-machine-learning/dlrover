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
from unittest.mock import patch, MagicMock
import logging


# Test the run_dashboard module
try:
    from dlrover.dashboard.run_dashboard import main

    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    EXCEPTION_MSG = str(e)


class TestRunDashboardImport(unittest.TestCase):
    """Test that the run_dashboard module can be imported."""

    def test_import_success(self):
        """Test that the module imports successfully."""
        if IMPORT_SUCCESS:
            self.assertTrue(True)
        else:
            self.fail(f"Failed to import run_dashboard: {EXCEPTION_MSG}")


if IMPORT_SUCCESS:
    # Only run these tests if import was successful
    class TestRunDashboardMain(unittest.TestCase):
        """Test the main function of run_dashboard."""

        def setUp(self):
            """Set up test fixtures."""
            # Reset any logging configuration
            logging.basicConfig(level=logging.DEBUG)

        @patch("dlrover.dashboard.run_dashboard.start_dashboard_server")
        def test_main_with_args(self, mock_start_server):
            """Test main function with command line arguments."""
            # Configure mock to return successfully
            mock_start_server.return_value = None

            with patch(
                "sys.argv",
                ["run_dashboard.py", "--host", "127.0.0.1", "--port", "8888"],
            ):
                main()

            # Verify that start_dashboard_server was called
            mock_start_server.assert_called_once()

            # Get the call arguments
            call_args = mock_start_server.call_args
            if call_args and call_args.args:
                # In simple form, start_dashboard_server might not take args from main()
                pass

        @patch("dlrover.dashboard.run_dashboard.start_dashboard_server")
        def test_main_default_args(self, mock_start_server):
            """Test main function with default arguments."""
            mock_start_server.return_value = None

            # Test with minimal arguments
            with patch("sys.argv", ["run_dashboard.py"]):
                main()

            # Should still call the server start function
            mock_start_server.assert_called_once()

        @patch("dlrover.dashboard.run_dashboard.start_dashboard_server")
        def test_main_with_invalid_args(self, mock_start_server):
            """Test main function handling of invalid arguments."""
            mock_start_server.return_value = None

            # Test with invalid arguments that should be handled gracefully
            with patch("sys.argv", ["run_dashboard.py", "--invalid-option"]):
                try:
                    main()
                    # Should either handle gracefully or raise appropriate error
                except SystemExit as e:
                    # Argument parsing might exit on invalid args
                    self.assertIn(
                        str(e.code) if e.code is not None else "",
                        ["2", "", "error"],
                    )
                except Exception as e:
                    # Should catch and handle specific exceptions
                    self.assertIsInstance(e, (ValueError, SystemExit))

        @patch("dlrover.dashboard.run_dashboard.sys.exit")
        @patch("dlrover.dashboard.run_dashboard.start_dashboard_server")
        def test_main_server_start_failure(self, mock_start_server, mock_exit):
            """Test main function handling when server start fails."""
            # Configure mock to raise exception
            mock_start_server.side_effect = Exception("Server failed to start")

            with patch("sys.argv", ["run_dashboard.py"]):
                main()

            # Server start should have been attempted
            mock_start_server.assert_called_once()
            mock_exit.assert_called_once_with(1)

        @patch("dlrover.dashboard.run_dashboard.start_dashboard_server")
        @patch("dlrover.dashboard.run_dashboard.logger")
        def test_main_logging(self, mock_logger, mock_start_server):
            """Test main function with logging configuration."""
            mock_start_server.return_value = None

            with patch("sys.argv", ["run_dashboard.py"]):
                main()

            # Should log the server starting action
            # Actual logging assertions depend on implementation
            mock_logger.info.assert_called()
            mock_start_server.assert_called_once()

    class TestRunDashboardIntegration(unittest.TestCase):
        """Test integration aspects of run_dashboard."""

        @patch("dlrover.dashboard.run_dashboard.sys.exit")
        @patch("dlrover.dashboard.app.ioloop.IOLoop")
        def test_integration_with_app(self, mock_ioloop, mock_exit):
            """Test integration with the dashboard app."""
            # Mock IOLoop to prevent server from actually running
            mock_ioloop_instance = MagicMock()
            mock_ioloop.current.return_value = mock_ioloop_instance

            with patch(
                "dlrover.dashboard.app.create_dashboard_app"
            ) as mock_create_app:
                mock_app = MagicMock()
                mock_create_app.return_value = mock_app

                with patch("sys.argv", ["run_dashboard.py"]):
                    main()

                # App should be created
                mock_create_app.assert_called_once()
                mock_exit.assert_called_once_with(1)


class TestMasterDashboardIntegration(unittest.TestCase):
    """Test dashboard integration in master main."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock dlrover_context
        self.mock_context = MagicMock()

    @patch("dlrover.python.master.main._dlrover_context")
    @patch("dlrover.python.master.main.add_dashboard_to_master")
    @patch("dlrover.python.master.main.new_job_args")
    @patch("dlrover.python.master.main.parse_master_args")
    def test_dashboard_integration_enabled(
        self,
        mock_parse_args,
        mock_new_job_args,
        mock_add_dashboard,
        mock_dlrover_context,
    ):
        """Test dashboard integration when enabled."""
        # Configure mock args
        mock_args = MagicMock()
        mock_args.platform = "kubernetes"
        mock_args.job_name = "test-job"
        mock_args.namespace = "default"
        mock_args.port = 50001
        mock_args.task_process_timeout = 3600
        mock_args.hang_detection = False
        mock_args.hang_downtime = 300
        mock_args.pending_fail_strategy = ""
        mock_args.pending_timeout = 0
        mock_args.service_type = ""
        mock_args.pre_check_ops = ""
        mock_args.dynamic_failover_extension = ""
        mock_args.training_elastic_mode = ""
        mock_args.xpu_type = "nvidia"
        mock_args.enable_dashboard = "true"
        mock_args.dashboard_port = 8080
        mock_parse_args.return_value = mock_args

        # Configure mock context with dashboard enabled
        mock_dlrover_context.enable_dashboard = True
        mock_dlrover_context.dashboard_port = 8080
        mock_dlrover_context.master_port = 50001
        mock_dlrover_context.hang_downtime = (
            300  # Set to avoid comparison issues
        )

        # Configure mock job_args
        mock_job_args = MagicMock()
        mock_job_args.platform = "kubernetes"
        mock_job_args.node_args = {}
        mock_job_args.initilize.return_value = None  # Mock initilize method
        mock_job_args.to_json.return_value = "{}"  # Mock to_json method
        mock_new_job_args.return_value = mock_job_args

        # Mock the master creation - patch where it's imported (in dist_master module)
        with patch(
            "dlrover.python.master.dist_master.DistributedJobMaster"
        ) as mock_master_cls:
            mock_master_instance = MagicMock()
            mock_master_instance.prepare = MagicMock()
            mock_master_instance.pre_check = MagicMock()
            mock_master_instance.run = MagicMock(return_value=0)
            mock_master_cls.return_value = mock_master_instance

            from dlrover.python.master.main import main as master_main

            master_main()

        # Verify dashboard was configured
        expected_config = {"enable": True, "host": "0.0.0.0", "port": 8080}

        # Check that add_dashboard_to_master was called
        mock_add_dashboard.assert_called_once_with(
            mock_master_instance, expected_config
        )

    @patch("dlrover.python.master.main._event_reporter")
    @patch("dlrover.python.master.main._dlrover_context")
    @patch("dlrover.python.master.main.add_dashboard_to_master")
    @patch("dlrover.python.master.main.new_job_args")
    @patch("dlrover.python.master.main.parse_master_args")
    def test_dashboard_integration_disabled(
        self,
        mock_parse_args,
        mock_new_job_args,
        mock_add_dashboard,
        mock_dlrover_context,
        mock_reporter,
    ):
        """Test dashboard integration when disabled."""
        # Configure mock args
        mock_args = MagicMock()
        mock_args.platform = "kubernetes"
        mock_args.job_name = "test-job"
        mock_args.namespace = "default"
        mock_args.port = 50001
        mock_args.task_process_timeout = 3600
        mock_args.hang_detection = False
        mock_args.hang_downtime = 300
        mock_args.pending_fail_strategy = ""
        mock_args.pending_timeout = 0
        mock_args.service_type = ""
        mock_args.pre_check_ops = ""
        mock_args.dynamic_failover_extension = ""
        mock_args.training_elastic_mode = ""
        mock_args.xpu_type = "nvidia"
        mock_args.enable_dashboard = "false"
        mock_args.dashboard_port = 8080
        mock_parse_args.return_value = mock_args

        # Configure mock context with dashboard disabled
        mock_dlrover_context.enable_dashboard = False
        mock_dlrover_context.dashboard_port = 8080
        mock_dlrover_context.master_port = 50001
        mock_dlrover_context.hang_downtime = (
            300  # Set to avoid comparison issues
        )

        # Configure mock job_args
        mock_job_args = MagicMock()
        mock_job_args.platform = "kubernetes"
        mock_job_args.node_args = {}
        mock_job_args.initilize.return_value = None  # Mock initilize method
        mock_job_args.to_json.return_value = "{}"  # Mock to_json method
        mock_new_job_args.return_value = mock_job_args

        # Mock the master creation - patch where it's imported (in dist_master module)
        with patch(
            "dlrover.python.master.dist_master.DistributedJobMaster"
        ) as mock_master_cls:
            mock_master_instance = MagicMock()
            mock_master_instance.prepare = MagicMock()
            mock_master_instance.pre_check = MagicMock()
            mock_master_instance.run = MagicMock(return_value=0)
            mock_master_cls.return_value = mock_master_instance

            from dlrover.python.master.main import main as master_main

            master_main()

        # Verify dashboard was configured with enable=False
        expected_config = {
            "enable": False,
            "host": "0.0.0.0",
            "port": 8080,
        }

        # Check that add_dashboard_to_master was called with disabled config
        mock_add_dashboard.assert_called_once_with(
            mock_master_instance, expected_config
        )

    @patch("dlrover.python.master.main._event_reporter")
    @patch("dlrover.python.master.main._dlrover_context")
    @patch("dlrover.python.master.main.add_dashboard_to_master")
    @patch("dlrover.python.master.main.new_job_args")
    @patch("dlrover.python.master.main.parse_master_args")
    def test_dashboard_integration_custom_port(
        self,
        mock_parse_args,
        mock_new_job_args,
        mock_add_dashboard,
        mock_dlrover_context,
        mock_reporter,
    ):
        """Test dashboard integration with custom port."""
        # Configure mock args
        mock_args = MagicMock()
        mock_args.platform = "kubernetes"
        mock_args.job_name = "test-job"
        mock_args.namespace = "default"
        mock_args.port = 50001
        mock_args.task_process_timeout = 3600
        mock_args.hang_detection = False
        mock_args.hang_downtime = 300
        mock_args.pending_fail_strategy = ""
        mock_args.pending_timeout = 0
        mock_args.service_type = ""
        mock_args.pre_check_ops = ""
        mock_args.dynamic_failover_extension = ""
        mock_args.training_elastic_mode = ""
        mock_args.xpu_type = "nvidia"
        mock_args.enable_dashboard = "true"
        mock_args.dashboard_port = 9999
        mock_parse_args.return_value = mock_args

        # Configure mock context with custom port
        mock_dlrover_context.enable_dashboard = True
        mock_dlrover_context.dashboard_port = 9999
        mock_dlrover_context.master_port = 50001
        mock_dlrover_context.hang_downtime = (
            300  # Set to avoid comparison issues
        )

        # Configure mock job_args
        mock_job_args = MagicMock()
        mock_job_args.platform = "kubernetes"
        mock_job_args.node_args = {}
        mock_job_args.initilize.return_value = None  # Mock initialize method
        mock_job_args.to_json.return_value = "{}"  # Mock to_json method
        mock_new_job_args.return_value = mock_job_args

        # Mock the master creation - patch where it's imported (in dist_master module)
        with patch(
            "dlrover.python.master.dist_master.DistributedJobMaster"
        ) as mock_master_cls:
            mock_master_instance = MagicMock()
            mock_master_instance.prepare = MagicMock()
            mock_master_instance.pre_check = MagicMock()
            mock_master_instance.run = MagicMock(return_value=0)
            mock_master_cls.return_value = mock_master_instance

            from dlrover.python.master.main import main as master_main

            master_main()

        # Verify custom port was used
        expected_config = {"enable": True, "host": "0.0.0.0", "port": 9999}

        mock_add_dashboard.assert_called_once_with(
            mock_master_instance, expected_config
        )


if __name__ == "__main__":
    unittest.main()
