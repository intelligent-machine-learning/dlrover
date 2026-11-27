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

"""Registration and discovery of recovery-aware training processes.

The local DLRover agent cannot assume that every training entrypoint supports
in-place recovery.  A recovery-aware process explicitly publishes a
``RecoveryCapability`` before the agent can send it a recovery request.  The
record is written with the atomic file helpers and is scoped by logical rank,
run id, and generation.

This module intentionally has no worker-launch hooks.  A training entrypoint
or a framework adapter calls :func:`register_process_capability` after it has
resolved its logical rank and before it initializes communication.  The agent
uses :func:`read_process_capabilities` to discover the processes that opted
in.  Ordinary jobs therefore do not need to import this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import uuid
from typing import Iterable, Optional, Sequence, Tuple

from dlrover.python.elastic_agent.in_place_recovery_files import (
    DEFAULT_MAX_JSON_BYTES,
    RecoveryFileError,
    atomic_write_json,
    ensure_recovery_layout,
    read_json,
    remove_file,
)
from dlrover.python.elastic_agent.in_place_recovery_protocol import (
    RecoveryCapability,
    RecoveryFileLayout,
    RecoveryProtocolError,
)


_CAPABILITY_NAME = re.compile(r"^rank-(?P<rank>[0-9]+)\.json$")


class CapabilityRegistrationError(RecoveryProtocolError):
    """Raised when a capability registration conflicts with a live record."""


def new_process_id(*, pid: Optional[int] = None) -> str:
    """Return a process identity suitable for a capability record.

    The PID alone is not an identity: it can be reused after a worker exits.
    Including a per-process UUID lets the agent distinguish a replacement
    process that happens to receive the same OS PID.
    """
    if pid is None:
        pid = os.getpid()
    if type(pid) is not int or pid <= 0:
        raise ValueError("pid must be a positive integer")
    hostname = socket.gethostname().strip() or "unknown-host"
    return f"{hostname}:{pid}:{uuid.uuid4().hex}"


def build_process_capability(
    *,
    run_id: str,
    generation: int,
    logical_rank: int,
    features: Iterable[str],
    pid: Optional[int] = None,
    process_id: Optional[str] = None,
) -> RecoveryCapability:
    """Build a capability for the current process.

    ``pid`` and ``process_id`` are optional for framework adapters.  When
    omitted, the current PID and a fresh host/PID/UUID identity are used.
    Callers that need retry-idempotence should keep and reuse the generated
    ``process_id`` for subsequent registration attempts.
    """
    if pid is None:
        pid = os.getpid()
    if process_id is None:
        process_id = new_process_id(pid=pid)
    return RecoveryCapability(
        run_id=run_id,
        generation=generation,
        logical_rank=logical_rank,
        pid=pid,
        process_id=process_id,
        features=tuple(features),
    )


def _validate_required_features(
    capability: RecoveryCapability,
    required_features: Sequence[str],
) -> None:
    if isinstance(required_features, (str, bytes)):
        raise CapabilityRegistrationError(
            "required_features must be a sequence of feature names"
        )
    required = tuple(required_features)
    if any(not isinstance(feature, str) or not feature for feature in required):
        raise CapabilityRegistrationError(
            "required_features must contain non-empty strings"
        )
    missing = sorted(set(required) - set(capability.features))
    if missing:
        raise CapabilityRegistrationError(
            "process is missing required recovery features: " + ", ".join(missing)
        )


def read_process_capability(
    layout: RecoveryFileLayout,
    logical_rank: int,
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> Optional[RecoveryCapability]:
    """Read one rank's registration, or ``None`` when it is absent."""
    path = layout.capabilities_path(logical_rank)
    payload = read_json(path, max_bytes=max_bytes)
    if payload is None:
        return None
    capability = RecoveryCapability.from_json(json.dumps(payload))
    if capability.logical_rank != logical_rank:
        raise CapabilityRegistrationError(
            f"capability path rank {logical_rank} does not match "
            f"payload rank {capability.logical_rank}"
        )
    return capability


def register_process_capability(
    layout: RecoveryFileLayout,
    capability: RecoveryCapability,
    *,
    required_features: Sequence[str] = (),
) -> Path:
    """Publish one process capability and reject stale/conflicting records.

    Registration is idempotent for the same process identity.  A newer
    generation may replace an older record for the same run and logical rank;
    a lower generation, a different run, or another process identity cannot
    overwrite the record.  This prevents a delayed worker from claiming a
    rank after a replacement has already registered it.
    """
    if not isinstance(capability, RecoveryCapability):
        raise TypeError("capability must be a RecoveryCapability")
    _validate_required_features(capability, required_features)
    ensure_recovery_layout(layout)

    existing = read_process_capability(layout, capability.logical_rank)
    if existing is not None:
        if existing.run_id != capability.run_id:
            raise CapabilityRegistrationError(
                "capability registration belongs to a different run"
            )
        if existing.generation > capability.generation:
            raise CapabilityRegistrationError(
                "stale capability generation cannot replace a newer registration"
            )
        if existing.generation == capability.generation:
            if (
                existing.process_id != capability.process_id
                or existing.pid != capability.pid
            ):
                raise CapabilityRegistrationError(
                    "another process is already registered for this rank and generation"
                )

    atomic_write_json(
        layout.capabilities_path(capability.logical_rank), capability.to_dict()
    )
    return layout.capabilities_path(capability.logical_rank)


def unregister_process_capability(
    layout: RecoveryFileLayout,
    capability: RecoveryCapability,
) -> bool:
    """Remove a registration only when it still belongs to this process.

    A stale process must not remove a replacement's registration.  In that
    case the function returns ``False`` and leaves the current record intact.
    """
    current = read_process_capability(layout, capability.logical_rank)
    if current is None:
        return False
    if current != capability:
        return False
    return remove_file(layout.capabilities_path(capability.logical_rank))


def read_process_capabilities(
    layout: RecoveryFileLayout,
    *,
    run_id: Optional[str] = None,
    generation: Optional[int] = None,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> Tuple[RecoveryCapability, ...]:
    """Discover all valid rank registrations in deterministic order."""
    ensure_recovery_layout(layout)
    try:
        entries = tuple((layout.root / "capabilities").iterdir())
    except OSError as exc:
        raise RecoveryFileError("cannot scan capability registrations") from exc

    capabilities = []
    for entry in sorted(entries, key=lambda item: item.name):
        match = _CAPABILITY_NAME.match(entry.name)
        if match is None:
            continue
        logical_rank = int(match.group("rank"))
        capability = read_process_capability(
            layout, logical_rank, max_bytes=max_bytes
        )
        if capability is None:
            continue
        if run_id is not None and capability.run_id != run_id:
            continue
        if generation is not None and capability.generation != generation:
            continue
        capabilities.append(capability)
    return tuple(capabilities)

