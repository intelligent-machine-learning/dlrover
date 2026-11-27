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

"""Atomic file operations for the local in-place recovery protocol.

The recovery protocol uses a node-local directory shared by one DLRover agent
and its training children.  Every published document is written to a private
temporary file, flushed, and atomically renamed into place.  Readers reject
symlinks, non-private files, oversized documents, and non-object JSON so a
crash or an unrelated file cannot be mistaken for a recovery command.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Mapping, Optional, Tuple, Union

from dlrover.python.elastic_agent.in_place_recovery_protocol import (
    RecoveryFileLayout,
    RecoveryProtocolError,
)


DEFAULT_FILE_MODE = 0o600
DEFAULT_DIRECTORY_MODE = 0o700
DEFAULT_MAX_JSON_BYTES = 1024 * 1024
DEFAULT_TEMP_MAX_AGE_SECONDS = 3600.0

_REQUEST_NAME = re.compile(r"^generation-(?P<generation>[0-9]{8})\.json$")
_TEMP_NAME = re.compile(r"^\.[^/]+\.tmp-[^/]+$")


class RecoveryFileError(OSError):
    """Raised when a recovery file is unsafe, malformed, or cannot be stored."""


def _private_mode(mode: int, *, directory: bool) -> int:
    required = 0o700 if directory else 0o600
    if type(mode) is not int or mode < 0 or mode & ~0o777:
        raise RecoveryFileError(f"invalid {'directory' if directory else 'file'} mode")
    if mode & 0o077 or mode & required != required:
        raise RecoveryFileError(
            f"{'directory' if directory else 'file'} mode must be owner-only"
        )
    return mode


def _lstat(path: Path):
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RecoveryFileError(f"cannot inspect recovery path {path}") from exc


def ensure_private_directory(
    path: Union[os.PathLike, str], *, mode: int = DEFAULT_DIRECTORY_MODE
) -> Path:
    """Create one private directory or validate an existing one."""
    path = Path(path)
    mode = _private_mode(mode, directory=True)
    try:
        path.mkdir(mode=mode, parents=False, exist_ok=True)
    except FileExistsError:
        pass
    except OSError as exc:
        raise RecoveryFileError(f"cannot create recovery directory {path}") from exc

    info = _lstat(path)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RecoveryFileError(f"recovery path is not a directory: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077 or stat.S_IMODE(info.st_mode) & 0o700 != 0o700:
        raise RecoveryFileError(f"recovery directory is not private: {path}")
    return path


def _ensure_directory_ancestor(path: Path) -> None:
    """Validate an existing ancestor without requiring private permissions.

    A private recovery directory is commonly rooted below ``/tmp`` or another
    shared parent.  The recovery files themselves remain private; only the
    existing ancestor needs to be a real directory and never a symlink.
    """
    info = _lstat(path)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RecoveryFileError(f"recovery directory ancestor is invalid: {path}")


def ensure_recovery_layout(
    layout: RecoveryFileLayout,
    *,
    mode: int = DEFAULT_DIRECTORY_MODE,
) -> RecoveryFileLayout:
    """Create and validate all directories used by the local protocol."""
    root = Path(layout.root)
    if root.parent == root:
        ensure_private_directory(root, mode=mode)
    else:
        # Create each component explicitly so an existing symlink in the path
        # is rejected rather than followed by mkdir(parents=True).
        missing = []
        current = root
        while _lstat(current) is None:
            missing.append(current)
            current = current.parent
        _ensure_directory_ancestor(current)
        for directory in reversed(missing):
            ensure_private_directory(directory, mode=mode)
    for directory in layout.directories[1:]:
        ensure_private_directory(directory, mode=mode)
    return layout


def _ensure_private_parent(path: Path) -> None:
    parent = path.parent
    info = _lstat(parent)
    if info is None or stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RecoveryFileError(f"recovery file parent is not a directory: {parent}")
    if stat.S_IMODE(info.st_mode) & 0o077 or stat.S_IMODE(info.st_mode) & 0o700 != 0o700:
        raise RecoveryFileError(f"recovery file parent is not private: {parent}")


def _ensure_private_existing_file(path: Path) -> os.stat_result:
    info = _lstat(path)
    if info is None:
        raise FileNotFoundError(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RecoveryFileError(f"recovery path is not a regular file: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077 or not stat.S_IMODE(info.st_mode) & 0o400:
        raise RecoveryFileError(f"recovery file is not private: {path}")
    return info


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RecoveryFileError(f"cannot open recovery directory {path}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise RecoveryFileError(f"cannot fsync recovery directory {path}") from exc
    finally:
        os.close(fd)


def atomic_write_bytes(
    path: Union[os.PathLike, str],
    payload: bytes,
    *,
    mode: int = DEFAULT_FILE_MODE,
) -> Path:
    """Atomically publish bytes with private permissions.

    The temporary file is created in the destination directory, so
    ``os.replace`` cannot cross filesystems.  Repeating the call for the same
    path is idempotent at the protocol level: the complete new document
    replaces the previous complete document and readers never observe a
    partial write.
    """
    path = Path(path)
    if not isinstance(payload, bytes):
        raise TypeError("recovery file payload must be bytes")
    mode = _private_mode(mode, directory=False)
    _ensure_private_parent(path)
    existing = _lstat(path)
    if existing is not None and stat.S_ISLNK(existing.st_mode):
        raise RecoveryFileError(f"refusing to replace symlink: {path}")

    fd = -1
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=str(path.parent)
        )
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
        return path
    except (OSError, ValueError) as exc:
        raise RecoveryFileError(f"cannot atomically publish recovery file {path}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def atomic_write_json(
    path: Union[os.PathLike, str],
    value: Mapping[str, Any],
    *,
    mode: int = DEFAULT_FILE_MODE,
) -> Path:
    """Serialize a JSON object canonically and publish it atomically."""
    if not isinstance(value, Mapping):
        raise TypeError("recovery JSON payload must be a mapping")
    try:
        payload = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecoveryProtocolError("recovery JSON payload is not serializable") from exc
    return atomic_write_bytes(path, payload, mode=mode)


def read_bytes(
    path: Union[os.PathLike, str],
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> Optional[bytes]:
    """Read one private file, returning ``None`` if it does not exist."""
    path = Path(path)
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    try:
        _ensure_private_existing_file(path)
    except FileNotFoundError:
        return None

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RecoveryFileError(f"cannot open recovery file {path}") from exc
    try:
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, max_bytes - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise RecoveryFileError(f"recovery file is too large: {path}")
        return b"".join(chunks)
    except OSError as exc:
        raise RecoveryFileError(f"cannot read recovery file {path}") from exc
    finally:
        os.close(fd)


def read_json(
    path: Union[os.PathLike, str],
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> Optional[dict]:
    """Read and validate one private JSON object."""
    payload = read_bytes(path, max_bytes=max_bytes)
    if payload is None:
        return None
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RecoveryProtocolError(f"recovery file is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryProtocolError(f"recovery file must contain a JSON object: {path}")
    return value


def remove_file(path: Union[os.PathLike, str]) -> bool:
    """Remove one regular recovery file or symlink, idempotently."""
    path = Path(path)
    info = _lstat(path)
    if info is None:
        return False
    if not stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        raise RecoveryFileError(f"refusing to remove non-file recovery path: {path}")
    try:
        path.unlink()
        _fsync_directory(path.parent)
    except OSError as exc:
        raise RecoveryFileError(f"cannot remove recovery file {path}") from exc
    return True


def _remove_old_temp_files(
    directory: Path,
    *,
    now: float,
    max_age_seconds: float,
    removed: list,
) -> None:
    try:
        entries = tuple(directory.iterdir())
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RecoveryFileError(f"cannot scan recovery directory {directory}") from exc
    for entry in entries:
        if not _TEMP_NAME.match(entry.name):
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError as exc:
            raise RecoveryFileError(f"cannot inspect temporary file {entry}") from exc
        if age >= max_age_seconds and remove_file(entry):
            removed.append(entry)


def cleanup_stale_files(
    layout: RecoveryFileLayout,
    *,
    keep_generation: int,
    temp_max_age_seconds: float = DEFAULT_TEMP_MAX_AGE_SECONDS,
) -> Tuple[Path, ...]:
    """Remove files belonging to generations older than ``keep_generation``.

    Malformed status, heartbeat, and capability files are retained for the
    agent to report as protocol errors.  Temporary files left by a crashed
    writer are removed after ``temp_max_age_seconds``.
    """
    if type(keep_generation) is not int or keep_generation < 0:
        raise ValueError("keep_generation must be a non-negative integer")
    if isinstance(temp_max_age_seconds, bool):
        raise ValueError("temp_max_age_seconds must be non-negative")
    try:
        temp_max_age = float(temp_max_age_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("temp_max_age_seconds must be a finite number") from exc
    if not math.isfinite(temp_max_age) or temp_max_age < 0:
        raise ValueError("temp_max_age_seconds must be a finite non-negative number")

    removed = []
    now = time.time()
    for directory in layout.directories[1:]:
        _remove_old_temp_files(
            directory,
            now=now,
            max_age_seconds=temp_max_age,
            removed=removed,
        )

    request_dir = layout.root / "requests"
    try:
        request_entries = tuple(request_dir.iterdir())
    except FileNotFoundError:
        request_entries = ()
    except OSError as exc:
        raise RecoveryFileError(f"cannot scan recovery directory {request_dir}") from exc
    for entry in request_entries:
        match = _REQUEST_NAME.match(entry.name)
        if match is None:
            continue
        generation = int(match.group("generation"))
        if generation < keep_generation and remove_file(entry):
            removed.append(entry)

    for directory_name in ("capabilities", "status", "heartbeats"):
        directory = layout.root / directory_name
        try:
            entries = tuple(directory.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RecoveryFileError(f"cannot scan recovery directory {directory}") from exc
        for entry in entries:
            if not entry.name.endswith(".json") or _TEMP_NAME.match(entry.name):
                continue
            payload = read_json(entry)
            if payload is None:
                continue
            generation = payload.get("generation")
            if type(generation) is int and generation < keep_generation:
                if remove_file(entry):
                    removed.append(entry)

    return tuple(sorted(removed, key=lambda path: str(path)))

