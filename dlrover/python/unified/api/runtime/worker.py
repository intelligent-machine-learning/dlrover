from typing import Protocol

from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.common.workload_base import (
    ActorInfo,
    JobInfo,
)


class Worker(Protocol):
    @property
    def job_info(self) -> JobInfo:
        """Get job information."""
        ...  # pragma: no cover

    @property
    def actor_info(self) -> ActorInfo:
        """Get actor information."""
        ...  # pragma: no cover


def current_worker() -> Worker:
    return BaseWorker.CURRENT
