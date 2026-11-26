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

"""Versioned local-file protocol for per-process in-place recovery.

The node-local DLRover agent and a recovery-capable training process exchange
small JSON documents through a job-private directory.  This module defines
the wire format and deterministic file names; it deliberately does not read,
write, watch, or delete files.  Those operations belong to the next protocol
layer so ordinary jobs never need to import a recovery listener.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple


PROTOCOL_VERSION = 1

CAPABILITY_TYPE = "capability"
REQUEST_TYPE = "recovery_request"
STATUS_TYPE = "recovery_status"
HEARTBEAT_TYPE = "heartbeat"

COMMAND_RECOVER = "recover"

STATUS_REGISTERED = "registered"
STATUS_OBSERVED = "observed"
STATUS_ABORTED = "aborted"
STATUS_QUIESCED = "quiesced"
STATUS_RECONFIGURED = "reconfigured"
STATUS_RESTORED = "restored"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

RECOVERY_STATES = (
    STATUS_REGISTERED,
    STATUS_OBSERVED,
    STATUS_ABORTED,
    STATUS_QUIESCED,
    STATUS_RECONFIGURED,
    STATUS_RESTORED,
    STATUS_READY,
    STATUS_FAILED,
)

# A training process may advertise only the features it implements.  The
# names are intentionally framework-neutral; Megatron can add its own
# communicator capabilities in a later adapter.
FEATURE_RECOVERY_COMMAND = "recovery_command"
FEATURE_PROCESS_GROUP_RECONFIGURE = "process_group_reconfigure"
FEATURE_CHECKPOINT_RESTORE = "checkpoint_restore"

_STATUS_TRANSITIONS = {
    STATUS_REGISTERED: frozenset((STATUS_REGISTERED, STATUS_OBSERVED, STATUS_FAILED)),
    STATUS_OBSERVED: frozenset((STATUS_OBSERVED, STATUS_ABORTED, STATUS_FAILED)),
    STATUS_ABORTED: frozenset((STATUS_ABORTED, STATUS_QUIESCED, STATUS_FAILED)),
    STATUS_QUIESCED: frozenset((STATUS_QUIESCED, STATUS_RECONFIGURED, STATUS_FAILED)),
    STATUS_RECONFIGURED: frozenset((STATUS_RECONFIGURED, STATUS_RESTORED, STATUS_FAILED)),
    STATUS_RESTORED: frozenset((STATUS_RESTORED, STATUS_READY, STATUS_FAILED)),
    STATUS_READY: frozenset((STATUS_READY,)),
    STATUS_FAILED: frozenset((STATUS_FAILED,)),
}


class RecoveryProtocolError(ValueError):
    """Raised when a local recovery document is malformed or incompatible."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _decode_json(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RecoveryProtocolError("payload is not UTF-8") from exc
    if not isinstance(payload, str):
        raise RecoveryProtocolError("payload must be a JSON string or bytes")
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise RecoveryProtocolError("payload is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RecoveryProtocolError("payload must contain a JSON object")
    return value


def _require_fields(value: Mapping[str, Any], expected: Iterable[str]) -> None:
    expected = set(expected)
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise RecoveryProtocolError("invalid fields: " + ", ".join(details))


def _require_protocol_header(value: Mapping[str, Any], message_type: str) -> None:
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise RecoveryProtocolError(
            f"unsupported recovery protocol version: {value.get('protocol_version')!r}"
        )
    if value.get("type") != message_type:
        raise RecoveryProtocolError(
            f"expected recovery document type {message_type!r}, "
            f"got {value.get('type')!r}"
        )


def _require_string(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise RecoveryProtocolError(f"{name} must be a non-empty string")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise RecoveryProtocolError(f"{name} contains a forbidden control character")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RecoveryProtocolError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise RecoveryProtocolError(f"{name} must be a positive integer")
    return value


def _normalize_features(features: Iterable[str]) -> Tuple[str, ...]:
    if isinstance(features, (str, bytes)):
        raise RecoveryProtocolError("features must be a sequence of strings")
    try:
        normalized = tuple(sorted(set(features)))
    except (TypeError, ValueError) as exc:
        raise RecoveryProtocolError("features must be a sequence of strings") from exc
    if not normalized or any(not isinstance(feature, str) or not feature for feature in normalized):
        raise RecoveryProtocolError("features must contain non-empty strings")
    for feature in normalized:
        _require_string(feature, "feature")
    return normalized


def _normalize_ranks(ranks: Iterable[int], name: str) -> Tuple[int, ...]:
    if isinstance(ranks, (str, bytes)):
        raise RecoveryProtocolError(f"{name} must be a sequence of ranks")
    try:
        normalized = tuple(sorted(ranks))
    except (TypeError, ValueError) as exc:
        raise RecoveryProtocolError(f"{name} must be a sequence of ranks") from exc
    if not normalized:
        raise RecoveryProtocolError(f"{name} must not be empty")
    if any(type(rank) is not int or rank < 0 for rank in normalized):
        raise RecoveryProtocolError(f"{name} must contain non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise RecoveryProtocolError(f"{name} must not contain duplicate ranks")
    return normalized


@dataclass(frozen=True)
class RecoveryReplacement:
    """One logical rank and the identity of its replacement process."""

    logical_rank: int
    replacement_id: str

    def __post_init__(self):
        _require_nonnegative_int(self.logical_rank, "logical_rank")
        _require_string(self.replacement_id, "replacement_id")

    def to_dict(self) -> dict:
        return {
            "logical_rank": self.logical_rank,
            "replacement_id": self.replacement_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RecoveryReplacement":
        if not isinstance(value, dict):
            raise RecoveryProtocolError("replacement must be a JSON object")
        _require_fields(value, ("logical_rank", "replacement_id"))
        return cls(value["logical_rank"], value["replacement_id"])


@dataclass(frozen=True)
class RecoveryCapability:
    """Capabilities registered by one recovery-aware training process."""

    run_id: str
    generation: int
    logical_rank: int
    pid: int
    process_id: str
    features: Tuple[str, ...]
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self):
        if self.protocol_version != PROTOCOL_VERSION:
            raise RecoveryProtocolError("unsupported recovery protocol version")
        _require_string(self.run_id, "run_id")
        _require_nonnegative_int(self.generation, "generation")
        _require_nonnegative_int(self.logical_rank, "logical_rank")
        _require_positive_int(self.pid, "pid")
        _require_string(self.process_id, "process_id")
        object.__setattr__(self, "features", _normalize_features(self.features))

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "type": CAPABILITY_TYPE,
            "run_id": self.run_id,
            "generation": self.generation,
            "logical_rank": self.logical_rank,
            "pid": self.pid,
            "process_id": self.process_id,
            "features": list(self.features),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: Any) -> "RecoveryCapability":
        value = _decode_json(payload)
        _require_fields(
            value,
            (
                "protocol_version",
                "type",
                "run_id",
                "generation",
                "logical_rank",
                "pid",
                "process_id",
                "features",
            ),
        )
        _require_protocol_header(value, CAPABILITY_TYPE)
        if not isinstance(value["features"], list):
            raise RecoveryProtocolError("features must be a JSON array")
        return cls(
            run_id=value["run_id"],
            generation=value["generation"],
            logical_rank=value["logical_rank"],
            pid=value["pid"],
            process_id=value["process_id"],
            features=tuple(value["features"]),
            protocol_version=value["protocol_version"],
        )


@dataclass(frozen=True)
class RecoveryRequest:
    """An immutable command written by a node agent for local processes."""

    command_id: str
    run_id: str
    generation: int
    failed_logical_ranks: Tuple[int, ...]
    replacements: Tuple[RecoveryReplacement, ...]
    checkpoint_id: str
    topology_digest: str
    deadline_ms: int
    protocol_version: int = PROTOCOL_VERSION
    command: str = COMMAND_RECOVER

    def __post_init__(self):
        if self.protocol_version != PROTOCOL_VERSION:
            raise RecoveryProtocolError("unsupported recovery protocol version")
        if self.command != COMMAND_RECOVER:
            raise RecoveryProtocolError(f"unsupported recovery command: {self.command!r}")
        _require_string(self.command_id, "command_id")
        _require_string(self.run_id, "run_id")
        _require_positive_int(self.generation, "generation")
        failed = _normalize_ranks(self.failed_logical_ranks, "failed_logical_ranks")
        if not self.replacements:
            raise RecoveryProtocolError("replacements must not be empty")
        replacements = tuple(self.replacements)
        if any(not isinstance(item, RecoveryReplacement) for item in replacements):
            raise RecoveryProtocolError("replacements contain an invalid entry")
        replacement_ranks = tuple(sorted(item.logical_rank for item in replacements))
        if len(set(replacement_ranks)) != len(replacement_ranks):
            raise RecoveryProtocolError("replacements contain duplicate logical ranks")
        if replacement_ranks != failed:
            raise RecoveryProtocolError(
                "failed_logical_ranks must match replacement logical ranks"
            )
        replacement_ids = tuple(item.replacement_id for item in replacements)
        if len(set(replacement_ids)) != len(replacement_ids):
            raise RecoveryProtocolError("replacement identities must be unique")
        _require_string(self.checkpoint_id, "checkpoint_id")
        _require_string(self.topology_digest, "topology_digest")
        _require_positive_int(self.deadline_ms, "deadline_ms")
        object.__setattr__(self, "failed_logical_ranks", failed)
        object.__setattr__(
            self,
            "replacements",
            tuple(sorted(replacements, key=lambda item: item.logical_rank)),
        )

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "type": REQUEST_TYPE,
            "command": self.command,
            "command_id": self.command_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "failed_logical_ranks": list(self.failed_logical_ranks),
            "replacements": [item.to_dict() for item in self.replacements],
            "checkpoint_id": self.checkpoint_id,
            "topology_digest": self.topology_digest,
            "deadline_ms": self.deadline_ms,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, payload: Any) -> "RecoveryRequest":
        value = _decode_json(payload)
        _require_fields(
            value,
            (
                "protocol_version",
                "type",
                "command",
                "command_id",
                "run_id",
                "generation",
                "failed_logical_ranks",
                "replacements",
                "checkpoint_id",
                "topology_digest",
                "deadline_ms",
            ),
        )
        _require_protocol_header(value, REQUEST_TYPE)
        if not isinstance(value["failed_logical_ranks"], list):
            raise RecoveryProtocolError("failed_logical_ranks must be a JSON array")
        if not isinstance(value["replacements"], list):
            raise RecoveryProtocolError("replacements must be a JSON array")
        return cls(
            command_id=value["command_id"],
            run_id=value["run_id"],
            generation=value["generation"],
            failed_logical_ranks=tuple(value["failed_logical_ranks"]),
            replacements=tuple(
                RecoveryReplacement.from_dict(item) for item in value["replacements"]
            ),
            checkpoint_id=value["checkpoint_id"],
            topology_digest=value["topology_digest"],
            deadline_ms=value["deadline_ms"],
            protocol_version=value["protocol_version"],
            command=value["command"],
        )


@dataclass(frozen=True)
class RecoveryStatus:
    """A monotonic status report written by one training process."""

    command_id: str
    run_id: str
    generation: int
    logical_rank: int
    pid: int
    process_id: str
    state: str
    timestamp_ms: int
    error: str = ""
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self):
        if self.protocol_version != PROTOCOL_VERSION:
            raise RecoveryProtocolError("unsupported recovery protocol version")
        _require_string(self.command_id, "command_id")
        _require_string(self.run_id, "run_id")
        _require_positive_int(self.generation, "generation")
        _require_nonnegative_int(self.logical_rank, "logical_rank")
        _require_positive_int(self.pid, "pid")
        _require_string(self.process_id, "process_id")
        if self.state not in RECOVERY_STATES:
            raise RecoveryProtocolError(f"unknown recovery status: {self.state!r}")
        _require_positive_int(self.timestamp_ms, "timestamp_ms")
        _require_string(self.error, "error", allow_empty=True)

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "type": STATUS_TYPE,
            "command_id": self.command_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "logical_rank": self.logical_rank,
            "pid": self.pid,
            "process_id": self.process_id,
            "state": self.state,
            "timestamp_ms": self.timestamp_ms,
            "error": self.error,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: Any) -> "RecoveryStatus":
        value = _decode_json(payload)
        _require_fields(
            value,
            (
                "protocol_version",
                "type",
                "command_id",
                "run_id",
                "generation",
                "logical_rank",
                "pid",
                "process_id",
                "state",
                "timestamp_ms",
                "error",
            ),
        )
        _require_protocol_header(value, STATUS_TYPE)
        return cls(
            command_id=value["command_id"],
            run_id=value["run_id"],
            generation=value["generation"],
            logical_rank=value["logical_rank"],
            pid=value["pid"],
            process_id=value["process_id"],
            state=value["state"],
            timestamp_ms=value["timestamp_ms"],
            error=value["error"],
            protocol_version=value["protocol_version"],
        )


@dataclass(frozen=True)
class RecoveryHeartbeat:
    """A liveness record written while a process handles a recovery request."""

    run_id: str
    generation: int
    logical_rank: int
    pid: int
    process_id: str
    timestamp_ms: int
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self):
        if self.protocol_version != PROTOCOL_VERSION:
            raise RecoveryProtocolError("unsupported recovery protocol version")
        _require_string(self.run_id, "run_id")
        _require_positive_int(self.generation, "generation")
        _require_nonnegative_int(self.logical_rank, "logical_rank")
        _require_positive_int(self.pid, "pid")
        _require_string(self.process_id, "process_id")
        _require_positive_int(self.timestamp_ms, "timestamp_ms")

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "type": HEARTBEAT_TYPE,
            "run_id": self.run_id,
            "generation": self.generation,
            "logical_rank": self.logical_rank,
            "pid": self.pid,
            "process_id": self.process_id,
            "timestamp_ms": self.timestamp_ms,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: Any) -> "RecoveryHeartbeat":
        value = _decode_json(payload)
        _require_fields(
            value,
            (
                "protocol_version",
                "type",
                "run_id",
                "generation",
                "logical_rank",
                "pid",
                "process_id",
                "timestamp_ms",
            ),
        )
        _require_protocol_header(value, HEARTBEAT_TYPE)
        return cls(
            run_id=value["run_id"],
            generation=value["generation"],
            logical_rank=value["logical_rank"],
            pid=value["pid"],
            process_id=value["process_id"],
            timestamp_ms=value["timestamp_ms"],
            protocol_version=value["protocol_version"],
        )


@dataclass(frozen=True)
class RecoveryFileLayout:
    """Deterministic node-local paths for the recovery documents."""

    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    def capabilities_path(self, logical_rank: int) -> Path:
        _require_nonnegative_int(logical_rank, "logical_rank")
        return self.root / "capabilities" / f"rank-{logical_rank:06d}.json"

    def request_path(self, generation: int) -> Path:
        _require_positive_int(generation, "generation")
        return self.root / "requests" / f"generation-{generation:08d}.json"

    def status_path(self, logical_rank: int) -> Path:
        _require_nonnegative_int(logical_rank, "logical_rank")
        return self.root / "status" / f"rank-{logical_rank:06d}.json"

    def heartbeat_path(self, logical_rank: int) -> Path:
        _require_nonnegative_int(logical_rank, "logical_rank")
        return self.root / "heartbeats" / f"rank-{logical_rank:06d}.json"

    @property
    def directories(self) -> Tuple[Path, ...]:
        return (
            self.root,
            self.root / "capabilities",
            self.root / "requests",
            self.root / "status",
            self.root / "heartbeats",
        )


def is_valid_status_transition(previous: Optional[str], current: str) -> bool:
    """Return whether a status update is valid for the local state machine."""
    if current not in RECOVERY_STATES:
        return False
    if previous is None:
        return current in (STATUS_REGISTERED, STATUS_OBSERVED, STATUS_FAILED)
    return current in _STATUS_TRANSITIONS.get(previous, frozenset())

