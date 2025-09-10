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

from dlrover.python.unified.common.actor_base import NodeInfo, WorkerStage

"""Remote Call define for controller to invoke backends."""


def setup() -> None:
    """Setup the actor.
    Always called when the actor is initialized or reconstructed, before other methods.
    """
    raise NotImplementedError("stub")  # pragma: no cover


def check_workers() -> None:
    """For SubMaster to check the status of workers."""
    raise NotImplementedError("stub")  # pragma: no cover


def get_stage() -> WorkerStage:
    """Get the stage of the role level backends."""
    raise NotImplementedError("stub")  # pragma: no cover


def start() -> None:
    """Start the role level backends."""
    raise NotImplementedError("stub")  # pragma: no cover


def restart_workers() -> None:
    """Restart the role level backends."""
    raise NotImplementedError("stub")  # pragma: no cover


def get_node_info() -> NodeInfo:
    """Get the ray node information."""
    raise NotImplementedError("stub")  # pragma: no cover
