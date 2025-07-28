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

from dlrover.python.unified.common.workload_base import WorkerStage

"""Remote Call define for rl backend. Shared between master and worker actors."""

# mypy: disable-error-code=empty-body


def status() -> WorkerStage:  # pragma: no cover
    """Get the status of the elastic training job."""
    ...


def get_master_addr() -> str:  # pragma: no cover
    """Get the master address."""
    ...


def setup_rl_workload(env_dict: dict) -> str:  # pragma: no cover
    """Setup basic env deps."""
    ...


def update_rl_workload_stage(worker_stage: WorkerStage):
    """Update workloads' worker-stage."""
    ...
