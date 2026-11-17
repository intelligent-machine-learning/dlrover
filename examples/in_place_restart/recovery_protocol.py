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
"""Small file-based control protocol used by the in-place restart demo."""

import json
import os
import secrets
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RecoveryCommand:
    """A request to roll back and reconfigure one recovery epoch."""

    recovery_epoch: int
    attempt_id: str
    failed_rank: int
    checkpoint_path: str
    checkpoint_step: int
    reconfigure_uuid: int
    world_size: int
    deadline_unix_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RecoveryCommand":
        command = cls(**value)
        if command.recovery_epoch <= 0:
            raise ValueError("recovery_epoch must be positive")
        if not command.attempt_id:
            raise ValueError("attempt_id must not be empty")
        if command.checkpoint_step < 0:
            raise ValueError("checkpoint_step must be nonnegative")
        if command.reconfigure_uuid <= 0:
            raise ValueError("reconfigure_uuid must be positive")
        return command


def fresh_reconfigure_uuid() -> int:
    """Return a positive signed-64-bit UUID for PyTorch reconfigure."""

    return secrets.randbits(62) + 1


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    """Atomically replace a JSON file after syncing it to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a complete JSON file, returning None while it is unavailable."""

    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_recovery_command(path: Path) -> Optional[RecoveryCommand]:
    value = read_json(path)
    if value is None:
        return None
    return RecoveryCommand.from_dict(value)


def wait_for_recovery_command(
    path: Path,
    timeout_seconds: float,
    minimum_epoch: int = 0,
) -> RecoveryCommand:
    """Wait for a recovery command newer than ``minimum_epoch``."""

    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            command = load_recovery_command(path)
        except (TypeError, ValueError) as error:
            last_error = error
            command = None
        if command is not None and command.recovery_epoch > minimum_epoch:
            return command
        time.sleep(0.2)

    detail = f": {last_error}" if last_error is not None else ""
    raise TimeoutError(
        f"timed out waiting for recovery command at {path}{detail}"
    )
