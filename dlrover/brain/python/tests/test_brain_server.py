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
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from dlrover.brain.python.server.server import BrainServer
from dlrover.brain.python.common.job import (
    NodeResource,
    NodeGroupResource,
    JobResource,
    JobOptimizePlan,
)


class TestBrainServer(unittest.TestCase):
    def setUp(self):
        self.brain_server = BrainServer()

        self.brain_server._optimizer_manager = MagicMock()
        self.brain_server._config_manager = MagicMock()

        self.client = TestClient(self.brain_server._server)

    def test_optimize_endpoint(self):
        """Test POST /optimize parses nested payloads and returns a rich JobOptimizePlan."""
        mock_plan = JobOptimizePlan(
            timestamp=1690000000,
            job_resource=JobResource(
                node_group_resources={
                    "worker": NodeGroupResource(
                        count=4,
                        resource=NodeResource(
                            memory=8192, cpu=8.0, gpu=1.0, gpu_type="A100"
                        ),
                    )
                }
            ),
        )
        self.brain_server._optimizer_manager.optimize.return_value = mock_plan

        payload = {
            "type": "training",
            "config": {
                "optimizer": "adam",
                "customized_config": {"learning_rate": "0.01"},
            },
            "job": {
                "uuid": "job-123",
                "cluster": "prod-cluster",
                "namespace": "dl-jobs",
                "user": "alice",
                "app": "training-app",
            },
        }

        response = self.client.post("/optimize", json=payload)

        self.assertEqual(response.status_code, 201)
        response_data = response.json()

        self.assertIn("job_opt_plan", response_data)
        plan_data = response_data["job_opt_plan"]

        self.assertEqual(plan_data["timestamp"], 1690000000)

        worker_group = plan_data["job_resource"]["node_group_resources"][
            "worker"
        ]
        self.assertEqual(worker_group["count"], 4)
        self.assertEqual(worker_group["resource"]["gpu_type"], "A100")
        self.assertEqual(worker_group["resource"]["cpu"], 8.0)

    def test_job_config_endpoint_success(self):
        """Test POST /job_config successfully returns configuration dictionary."""
        fake_config = MagicMock()
        expected_configs = {"memory": "16G", "cpu": "8"}

        # UPDATED: We now mock get_config_values instead of convert_to_dict
        fake_config.get_config_values.return_value = expected_configs

        self.brain_server._config_manager.get_job_config.return_value = (
            fake_config
        )

        payload = {
            "type": "inference",
            "job": {"uuid": "job-456", "user": "bob"},
        }

        response = self.client.post("/job_config", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_configs"], expected_configs)

    def test_job_config_endpoint_not_found(self):
        """Test POST /job_config safely handles a None return and outputs {}."""
        self.brain_server._config_manager.get_job_config.return_value = None

        payload = {"type": "test", "job": {"uuid": "unknown"}}

        response = self.client.post("/job_config", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_configs"], {})


if __name__ == "__main__":
    unittest.main()
