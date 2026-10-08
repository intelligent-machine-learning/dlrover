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

"""Shared protocol of the urgent-save marker file.

The protocol lets an operator ask a training job to save a checkpoint as
soon as possible before the faulty node is isolated (see the design doc
"DLRover proactive checkpoint"). Its full state is the existence of a
marker file in the shared save dir:

- The master REST API (`POST /api/v1/urgent/save`, only enabled with
  `--mcp-svc`) publishes a request on the KV bus; the rank-0 node agent
  writes ``{save_dir}/.urgent_save.marker`` (atomically) next to the
  checkpoint directories.
- The training loop (atorch) checks the marker on every step boundary,
  collectively, and saves a checkpoint through the regular save path.
- When the save is fully finalized, the training world rank-0 removes the
  marker; the megatron tracker file (``latest_checkpointed_iteration.txt``)
  is written right before the removal, so the tracker flip is the success
  signal the agent observes and reports back on the KV bus.
- The node check and the hang diagnosis of dlrover are NOT part of this
  protocol: they keep running independently and never trigger it.

This module holds the parts shared by the master (via the dlrover-addon
mcp service) and the rank-0 node agent, so no proto/IDL changes are
needed: the KV bus just carries small JSON payloads.
"""

import json
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

MARKER_FILENAME = ".urgent_save.marker"
TRACKER_FILENAME = "latest_checkpointed_iteration.txt"

# KV bus keys, backed by MasterServicer's KVStoreService. The values are
# UTF-8 JSON strings; every command carries a unique ``ts`` so the agent can
# de-duplicate and the master can await the matching result.
KV_CMD_KEY = "urgent_save:cmd"  # master -> rank-0 agent: latest save request
KV_CLEANUP_KEY = "urgent_save:cleanup"  # master -> rank-0 agent: clear stale markers on startup
KV_RESULT_KEY_PREFIX = (
    "urgent_save:result:"  # agent -> master: keyed by the cmd ts
)

ACTION_SAVE = "save"
ACTION_CLEANUP = "cleanup"

STATUS_RECEIVED = "received"  # the agent has written the marker
STATUS_SAVED = "saved"  # the tracker flipped: the checkpoint is complete
STATUS_ERROR = "error"  # the agent failed to write/handle the marker


def marker_path(save_dir: str) -> str:
    return os.path.join(save_dir, MARKER_FILENAME)


def tracker_path(save_dir: str) -> str:
    return os.path.join(save_dir, TRACKER_FILENAME)


def result_key(cmd_ts: float) -> str:
    return KV_RESULT_KEY_PREFIX + str(cmd_ts)


def build_cmd(cmd_ts: float, save_dir: str, reason: str = "") -> bytes:
    """Serialize a save request published on KV_CMD_KEY."""
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "ts": cmd_ts,
            "action": ACTION_SAVE,
            "save_dir": save_dir,
            "reason": reason,
        }
    ).encode("utf-8")


def build_cleanup_cmd(cmd_ts: float, save_dir: str) -> bytes:
    """Serialize a cleanup request published on KV_CLEANUP_KEY."""
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "ts": cmd_ts,
            "action": ACTION_CLEANUP,
            "save_dir": save_dir,
        }
    ).encode("utf-8")


def build_result(cmd_ts: float, status: str, message: str = "") -> bytes:
    """Serialize a result published on the per-request result key."""
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "cmd_ts": cmd_ts,
            "status": status,
            "message": message,
        }
    ).encode("utf-8")


def parse_payload(value) -> Optional[dict]:
    """Decode a KV payload into a dict; None for empty/invalid values."""
    if not value:
        return None
    try:
        payload = json.loads(value.decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except (ValueError, AttributeError, UnicodeDecodeError):
        return None


def atomic_write_marker(save_dir: str, payload: bytes) -> str:
    """Create the marker atomically (write a tmp file, fsync and rename),
    so the readers only ever observe the full content or nothing.
    """
    marker = marker_path(save_dir)
    tmp = "{}.tmp.{}".format(marker, os.getpid())
    with open(tmp, "wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, marker)
    return marker


def read_tracker(save_dir: str) -> Tuple[Optional[int], float]:
    """Return ``(iteration, mtime)`` of the tracker file.

    ``iteration`` is None when the tracker is absent or unreadable, e.g. no
    checkpoint was saved yet or the save dir is not mounted. The tracker is
    written by the training rank-0 at the end of the finalize chain, so a
    growing iteration means the save truly completed.
    """
    path = tracker_path(save_dir)
    try:
        with open(path, "r") as f:
            iteration = int(f.read().strip())
        return iteration, os.path.getmtime(path)
    except (OSError, ValueError):
        return None, 0.0


def remove_marker(save_dir: str) -> bool:
    """Idempotently remove the marker; False when it was already gone."""
    try:
        os.unlink(marker_path(save_dir))
        return True
    except FileNotFoundError:
        return False
    except OSError as e:
        logger.warning("Fail to remove the marker in %s: %s", save_dir, e)
        return False


_master_kv_store = None


def register_master_kv_store(kv_store):
    """Remember the master KVStoreService so code running in the master
    process (e.g. the mcp REST handler of dlrover-addon) can publish urgent
    save requests on the same bus the agents poll over gRPC.
    """
    global _master_kv_store
    _master_kv_store = kv_store


def get_master_kv_store():
    """Return the registered master KV store, or None before the master
    servicer is constructed."""
    return _master_kv_store
