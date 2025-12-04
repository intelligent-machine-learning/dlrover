import unittest
from unittest.mock import MagicMock
import requests
from dlrover.brain.python.client.client import BrainClient
from dlrover.brain.python.common.http_schemas import (
    OptimizeRequest,
    OptimizeResponse,
    Response,
)
from dlrover.brain.python.common.optimize import (
    JobMeta,
    JobOptimizePlan,
)


class TestBrainClient(unittest.TestCase):
    def setUp(self):
        """Runs before every test method"""
        self.base_url = "http://api.brain.local"
        self.client = BrainClient(self.base_url)

        # Create a mock session to replace the real requests.Session
        self.mock_session = MagicMock()
        # Inject our mock into the client
        self.client.session = self.mock_session

    def test_init_creates_robust_session(self):
        """Verify the session is configured with retries."""
        real_client = BrainClient("http://test.com")

        # Check if HTTPAdapter is mounted (this confirms retries are configured)
        adapter = real_client.session.get_adapter("http://")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.max_retries.total, 3)
        self.assertEqual(adapter.max_retries.backoff_factor, 1)

    def test_optimize_success(self):
        job_uid = "job-uuid"
        job = JobMeta(uuid=job_uid)
        req_data = OptimizeRequest(job=job)

        opt_response = OptimizeResponse(
            response=Response(success=True),
            job_opt_plan=JobOptimizePlan(
                job_meta=job,
            ),
        )

        expected_resp_json = opt_response.model_dump()

        # 2. Configure the Mock Response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_resp_json
        self.mock_session.post.return_value = mock_response

        # 3. Execution
        result = self.client.optimize(req_data)

        # 4. Assertions
        # Verify the result is the correct Pydantic model
        self.assertIsInstance(result, OptimizeResponse)
        self.assertEqual(result.job_opt_plan.job_meta.uuid, job_uid)

        # Verify the client called the correct URL with correct JSON
        self.mock_session.post.assert_called_once_with(
            "http://api.brain.local/optimize",
            json=req_data.model_dump(),
        )

    def test_optimize_error(self):
        job_uid = "job-uuid"
        job = JobMeta(uuid=job_uid)
        req_data = OptimizeRequest(job=job)

        # 2. Configure Mock to simulate a 500 error
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        # raise_for_status() should raise HTTPError when called
        error = requests.exceptions.HTTPError("500 Error", response=mock_response)
        mock_response.raise_for_status.side_effect = error

        self.mock_session.post.return_value = mock_response
        resp = self.client.optimize(req_data)
        self.assertIsNone(resp)

        self.mock_session.post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        resp = self.client.optimize(req_data)
        self.assertIsNone(resp)

    def test_optimize_validation_error(self):
        job_uid = "job-uuid"
        job = JobMeta(uuid=job_uid)
        req_data = OptimizeRequest(job=job)

        # Server returns valid JSON, but missing required field 'plan_id'
        bad_json = {"success": True}  # Missing 'plan_id'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = bad_json
        self.mock_session.post.return_value = mock_response

        self.mock_session.post.side_effect = Exception("Exception!!!")
        resp = self.client.optimize(req_data)
        self.assertIsNone(resp)


if __name__ == '__main__':
    unittest.main()
