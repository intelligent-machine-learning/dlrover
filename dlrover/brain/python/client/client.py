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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dlrover.brain.python.common.http_schemas import (
    OptimizeRequest,
    OptimizeResponse,
    JobConfigRequest,
    JobConfigResponse,
)
from dlrover.brain.python.common.log import default_logger as logger
from typing import Optional


class BrainClient:
    """
    An HTTP client for interacting with the DLRover Brain Server.

    This client provides robust communication with the Brain Server by
    utilizing connection pooling and automatic retries for transient
    network or server-side infrastructure errors (e.g., 502, 503, 504).
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = self._create_robust_session()

    def _create_robust_session(self) -> requests.Session:
        """
        Creates a session with automatic retries for transient errors.
        """
        session = requests.Session()

        # Retry strategy:
        # - total=3: Try 3 times total
        # - backoff_factor=1: Wait 1s, then 2s, then 4s between retries
        # - status_forcelist: Retry on 502, 503, 504 (Server errors)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def optimize(self, request: OptimizeRequest) -> Optional[OptimizeResponse]:
        """
        Get an optimize plan for a training job from the Brain.
        """
        url = f"{self.base_url}/optimize"

        try:
            response = self.session.post(url, json=request.model_dump())
            response.raise_for_status()

            return OptimizeResponse(**response.json())
        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP Error: {e.response.status_code} - {e.response.text}"
            )
            return None
        except requests.exceptions.ConnectionError:
            logger.info("Failed to connect to the Brain server.")
            return None
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")
            return None

    def get_config(
        self, request: JobConfigRequest
    ) -> Optional[JobConfigResponse]:
        """
        Retrieves the job configuration from the Brain.
        """
        url = f"{self.base_url}/job_config"

        try:
            response = self.session.post(url, json=request.model_dump())
            response.raise_for_status()

            return JobConfigResponse(**response.json())
        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP Error fetching config: {e.response.status_code} - {e.response.text}"
            )
            return None
        except requests.exceptions.ConnectionError:
            logger.info(
                "Failed to connect to the Brain server to fetch config."
            )
            return None
        except Exception as e:
            logger.info(f"An unexpected error occurred fetching config: {e}")
            return None
