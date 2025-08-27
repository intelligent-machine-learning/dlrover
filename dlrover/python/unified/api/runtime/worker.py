# Copyright 2025 The DLRover Authors. All rights reserved.
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

from typing import Protocol

from dlrover.python.unified.common.actor_base import (
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
    """Get the information for the current worker."""

    from dlrover.python.unified.backend.common.base_worker import BaseWorker

    return BaseWorker.CURRENT
