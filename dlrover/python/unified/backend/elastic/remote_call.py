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


"""Remote Call define for elastic backend. Shared between master and worker actors."""

# mypy: disable-error-code=empty-body

from dlrover.python.unified.util.actor_helper import actor_call


@actor_call
def get_master_addr() -> str:
    """Get the master address."""
    raise NotImplementedError("stub")  # pragma: no cover


def setup_torch_process_group(
    master_addr: str, world_size: int, rank: int, only_envs: bool = False
) -> None:
    """Setup the torch process group.

    If `only_envs` is True, only set the environment variables MASTER_ADDR and MASTER_PORT.
    """
    raise NotImplementedError("stub")  # pragma: no cover


def destroy_torch_process_group() -> None:
    """Destroy the torch process group."""
    raise NotImplementedError("stub")  # pragma: no cover


def get_ray_node_id() -> str:
    """Get the Ray node ID."""
    raise NotImplementedError("stub")  # pragma: no cover


def run_network_check() -> float:
    """Run network check before starting the job.
    Returns the time taken for the check.
    """
    raise NotImplementedError("stub")  # pragma: no cover


def start_elastic_job() -> None:
    """Start the elastic training job."""
    raise NotImplementedError("stub")  # pragma: no cover
