import unittest
from unittest.mock import patch, MagicMock
from kubernetes import client
from dlrover.brain.python.platform.k8s.configmap import ConfigMapWatcher


class TestConfigMapWatcher(unittest.TestCase):

    @patch("dlrover.brain.python.platform.k8s.configmap.watch.Watch")
    @patch("dlrover.brain.python.platform.k8s.configmap.client.CoreV1Api")
    @patch("dlrover.brain.python.platform.k8s.configmap.config")
    def test_watch_calls_callback_on_modified(self, mock_config, mock_api_cls, mock_watch_cls):
        """
        Scenario: The watcher receives a 'MODIFIED' event.
        Goal: Verify that 'on_update_callback' is called with the correct data.
        """
        # --- SETUP ---

        # 1. Create a Fake Callback
        # We use MagicMock so we can verify if it was called later
        mock_callback = MagicMock()

        # 2. Create a Fake Event
        fake_data = {"learning_rate": "0.01", "batch_size": "32"}
        fake_event = {
            'type': 'MODIFIED',
            'object': MagicMock(data=fake_data, metadata=MagicMock(resource_version="500"))
        }

        # 3. Setup the Watcher Mock to BREAK the infinite loop
        # - Iteration 1: Yields [fake_event] (The loop runs once)
        # - Iteration 2: Raises KeyboardInterrupt (The loop crashes so the test finishes)
        mock_watch_instance = mock_watch_cls.return_value
        mock_watch_instance.stream.side_effect = [
            [fake_event],      # First call returns the list of events
            KeyboardInterrupt  # Second call raises exception to stop the "while True"
        ]

        # 4. Initialize the Class
        watcher = ConfigMapWatcher("default", "brain-config", mock_callback)

        # --- EXECUTION ---
        try:
            watcher.watch()
        except KeyboardInterrupt:
            pass  # Expected behavior to end the test

        # --- ASSERTIONS ---

        # A. Verify K8s config was loaded
        # It should try incluster first, or kube_config if that fails.
        assert mock_config.load_incluster_config.called or mock_config.load_kube_config.called

        # B. Verify correct API method was passed to stream()
        # The first argument to w.stream MUST be the list function
        mock_api_instance = mock_api_cls.return_value

        # Get the args passed to stream()
        call_args = mock_watch_instance.stream.call_args
        passed_func = call_args[0][0]  # First arg of the call

        self.assertEqual(passed_func, mock_api_instance.list_namespaced_config_map)

        # C. Verify the callback received the data
        mock_callback.assert_called_once_with(fake_data)

    @patch("dlrover.brain.python.platform.k8s.configmap.time.sleep")
    @patch("dlrover.brain.python.platform.k8s.configmap.watch.Watch")
    @patch("dlrover.brain.python.platform.k8s.configmap.config")
    def test_retry_on_connection_error(self, mock_config, mock_watch_cls, mock_sleep):
        """
        Scenario: The K8s API raises an exception (e.g. 410 Gone or Network Error).
        Goal: Verify the code sleeps and retries.
        """
        # Setup
        mock_callback = MagicMock()
        mock_watch_instance = mock_watch_cls.return_value

        # Logic:
        # 1. First call -> Raises Exception ("Boom")
        # 2. Code catches it -> calls sleep(5) -> Loops again
        # 3. Second call -> Raises KeyboardInterrupt -> Test ends
        mock_watch_instance.stream.side_effect = [
            Exception("Network Error"),
            KeyboardInterrupt
        ]

        watcher = ConfigMapWatcher("default", "brain-config", mock_callback)

        try:
            watcher.watch()
        except KeyboardInterrupt:
            pass

        # Assert that sleep(5) was called exactly once
        mock_sleep.assert_called_with(5)
        print("SUCCESS: Retry logic triggered sleep(5).")


if __name__ == '__main__':
    unittest.main()
