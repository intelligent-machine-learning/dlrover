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

"""Remote Call define for elastic backend. Shared between master and worker actors."""

# mypy: disable-error-code=empty-body


def status() -> WorkerStage:  # pragma: no cover
    """Get the status of the elastic training job."""
    ...


def get_master_addr() -> str:  # pragma: no cover
    """Get the master address."""
    ...


def setup_torch_process_group(
    master_addr: str, world_size: int, rank: int
) -> None:  # pragma: no cover
    """Setup the torch process group."""
    ...


def destroy_torch_process_group() -> None:  # pragma: no cover
    """Destroy the torch process group."""
    ...


def get_ray_node_id() -> str:  # pragma: no cover
    """Get the Ray node ID."""
    ...


def run_network_check() -> float:  # pragma: no cover
    """Run network check before starting the job.
    Returns the time taken for the check.
    """
    ...


def start_elastic_job() -> None:  # pragma: no cover
    """Start the elastic training job."""
    ...
