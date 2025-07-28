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


def status() -> WorkerStage:
    """Get the status of the elastic training job."""
    raise NotImplementedError("stub")  # pragma: no cover


def get_master_addr() -> str:
    """Get the master address."""
    raise NotImplementedError("stub")  # pragma: no cover


def setup_rl_workload(env_dict: dict) -> str:
    """Setup basic env deps."""
    raise NotImplementedError("stub")  # pragma: no cover


def update_rl_workload_stage(worker_stage: WorkerStage):
    """Update workloads' worker-stage."""
    raise NotImplementedError("stub")  # pragma: no cover
