import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from dlrover.brain.python.server.server import BrainServer
from dlrover.brain.python.common.optimize import (
    NodeResource,
    NodeGroupResource,
    JobResource,
    JobOptimizePlan,
    JobMeta,
)


class TestBrainServer(unittest.TestCase):
    def setUp(self):
        self.brain_server = BrainServer()
        self.brain_server.optimizer_manager = MagicMock()
        self.mock_manager_instance = self.brain_server.optimizer_manager
        self.brain_server.register_routes()
        self.client = TestClient(self.brain_server.server)

    def test_optimize_endpoint_returns_complex_plan(self):
        req_payload = {
            "type": "standard",
            "job": {
                "uuid": "job-uuid-123",
                "cluster": "prod-cluster",
                "namespace": "default"
            },
            "config": {
                "optimizer": "genetic_algo",
                "customized_config": {"pop_size": "50"}
            }
        }

        # --- B. Setup Expected "Internal" Result ---
        # We manually construct the complex objects the OptimizerManager would return

        # 1. Define resources for a "worker" group
        worker_resource = NodeResource(
            cpu=8.0,
            memory=32000,
            gpu=1.0,
            gpu_type="nvidia-v100",
            priority=1
        )

        worker_group = NodeGroupResource(
            count=4,
            resource=worker_resource
        )

        # 2. Put it into the JobResource
        # Note: Pydantic expects a dict for 'node_group_resources'
        job_resources = JobResource(
            node_group_resources={"worker": worker_group}
        )

        # 3. Create the final Plan object
        fake_plan = JobOptimizePlan(
            timestamp=1700000000,
            job_meta=JobMeta(uuid="job-uuid-123"),
            job_resource=job_resources
        )

        # --- C. Configure Mock ---
        # When manager.optimize() is called, return our complex object
        self.mock_manager_instance.optimize.return_value = fake_plan

        # --- D. Execution ---
        # Assuming your server uses GET. (Switch to .post() if you updated the server)
        response = self.client.request("GET", "/optimize", json=req_payload)

        # --- E. Assertions ---
        self.assertEqual(response.status_code, 201)

        json_data = response.json()

        # 1. Check the Wrapper (OptimizeResponse)
        self.assertIn("job_opt_plan", json_data)

        # 2. Check the Deeply Nested Data
        plan_data = json_data["job_opt_plan"]

        # Check Timestamp
        self.assertEqual(plan_data["timestamp"], 1700000000)

        # Check Node Resources (Drill down: Plan -> JobResource -> Dict -> NodeGroup -> NodeResource)
        worker_data = plan_data["job_resource"]["node_group_resources"]["worker"]

        self.assertEqual(worker_data["count"], 4)
        self.assertEqual(worker_data["resource"]["cpu"], 8.0)
        self.assertEqual(worker_data["resource"]["gpu_type"], "nvidia-v100")

        # 3. Verify arguments passed to the manager
        self.mock_manager_instance.optimize.assert_called_once()
        args, _ = self.mock_manager_instance.optimize.call_args

        # Verify the Input Pydantic models were parsed correctly
        self.assertEqual(args[0].uuid, "job-uuid-123")         # JobMeta
        self.assertEqual(args[1].optimizer, "genetic_algo")


if __name__ == '__main__':
    unittest.main()