import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dlrover.brain.python.common.http_schemas import (
    OptimizeRequest,
    OptimizeResponse,
)
from dlrover.brain.python.common.log import default_logger as logger
from typing import Optional


class BrainClient:
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
        # - status_forcelist: Retry on 500, 502, 503, 504 (Server errors)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def optimize(self, request: OptimizeRequest) -> Optional[OptimizeResponse]:
        url = f"{self.base_url}/optimize"

        try:
            response = self.session.post(url, json=request.model_dump())
            response.raise_for_status()

            return OptimizeResponse(**response.json())
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            return None
        except requests.exceptions.ConnectionError:
            logger.info("Failed to connect to the Brain server.")
            return None
        except Exception as e:
            logger.info(f"An unexpected error occurred: {e}")
            return None

