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

"""The rank-0 node agent side of the urgent-save marker protocol.

The listener polls the master KV bus (``MasterClient.kv_store_get/set``,
backed by the master ``KVStoreService``) and runs on the node whose agent
``node_rank == 0`` - that node hosts the training world rank 0, the only
rank that writes the megatron tracker and the only rank allowed to create
or delete the marker by protocol.

Requests (see ``dlrover.python.common.urgent_save``):

- ``urgent_save:cmd`` - publish the marker into the given save dir, then
  observe the tracker until it flips past the pre-write snapshot (that is
  the moment the checkpoint is fully saved and finalized by the training
  side, which also unlinks the marker) and publish the result.
- ``urgent_save:cleanup`` - remove a stale marker, used when the master
  starts up so a request left by a previous run cannot fire again.

There is no timeout and no detection logic here by design: the observer
waits until the tracker flips; if the training hangs or is relaunched, the
regular dlrover mechanisms (node check / hang diagnosis, independent of
this protocol) take over and the restart clears everything.
"""

import json
import logging
import os
import threading
import time

from dlrover.python.common.urgent_save import (
    ACTION_CLEANUP,
    ACTION_SAVE,
    KV_CLEANUP_KEY,
    KV_CMD_KEY,
    PROTOCOL_VERSION,
    STATUS_ERROR,
    STATUS_RECEIVED,
    STATUS_SAVED,
    atomic_write_marker,
    build_result,
    marker_path,
    parse_payload,
    read_tracker,
    remove_marker,
    result_key,
    tracker_path,
)

logger = logging.getLogger(__name__)


class UrgentSaveListener(threading.Thread):
    """Poll the KV bus and serve urgent-save requests on the rank-0 node."""

    def __init__(self, master_client, poll_interval_seconds=5.0):
        super().__init__(name="urgent-save-listener", daemon=True)
        self._client = master_client
        self._poll_interval = poll_interval_seconds
        self._seen_cmd_ts = 0.0
        self._seen_cleanup_ts = 0.0
        self._node_name = os.environ.get("POD_NAME") or os.environ.get(
            "HOSTNAME", ""
        )

    def run(self):
        logger.info(
            "Urgent-save listener starts on node %s (poll every %.1fs).",
            self._node_name,
            self._poll_interval,
        )
        while True:
            try:
                self._poll_once()
            except Exception as e:  # noqa: BLE001 - keep the listener alive
                logger.error("Urgent-save listener iteration failed: %s", e)
            time.sleep(self._poll_interval)

    def _poll_once(self):
        self._maybe_cleanup()
        self._maybe_save()

    def _maybe_cleanup(self):
        """Drop a stale marker when the master restarts with a cleanup cmd."""
        cmd = parse_payload(self._client.kv_store_get(KV_CLEANUP_KEY))
        if cmd is None or cmd.get("action") != ACTION_CLEANUP:
            return
        cmd_ts = cmd.get("ts", 0.0)
        if cmd_ts <= self._seen_cleanup_ts:
            return
        self._seen_cleanup_ts = cmd_ts
        save_dir = cmd.get("save_dir")
        if not save_dir:
            return
        if remove_marker(save_dir):
            logger.info(
                "Urgent-save: stale marker removed at startup: %s",
                marker_path(save_dir),
            )

    def _maybe_save(self):
        """Handle a new save request: snapshot the tracker, write the
        marker, report receipt and observe the tracker flip."""
        cmd = parse_payload(self._client.kv_store_get(KV_CMD_KEY))
        if cmd is None or cmd.get("action") != ACTION_SAVE:
            return
        cmd_ts = cmd.get("ts", 0.0)
        if cmd_ts <= self._seen_cmd_ts:
            return
        self._seen_cmd_ts = cmd_ts
        save_dir = cmd.get("save_dir")
        reason = cmd.get("reason", "")
        if not save_dir:
            self._publish(
                cmd_ts, STATUS_ERROR, "'save_dir' missing in the request"
            )
            return

        try:
            base_iteration, _base_mtime = read_tracker(save_dir)
            marker = atomic_write_marker(
                save_dir,
                json.dumps(
                    {
                        "v": PROTOCOL_VERSION,
                        "ts": cmd_ts,
                        "reason": reason,
                        "writer": "dlrover-agent@{}".format(self._node_name),
                    }
                ).encode("utf-8"),
            )
            self._publish(
                cmd_ts, STATUS_RECEIVED, "marker written at {}".format(marker)
            )
        except Exception as e:  # noqa: BLE001 - report instead of dying
            logger.error(
                "Urgent-save: fail to write the marker in %s: %s", save_dir, e
            )
            self._publish(cmd_ts, STATUS_ERROR, str(e))
            return

        observer = threading.Thread(
            target=self._observe,
            args=(cmd_ts, save_dir, base_iteration, marker),
            name="urgent-save-observer-{}".format(cmd_ts),
            daemon=True,
        )
        observer.start()

    def _observe(self, cmd_ts, save_dir, base_iteration, marker):
        """Wait until the tracker flips past the snapshot taken before the
        marker write - i.e. the training saved and finalized a checkpoint -
        then report success. No timeout by protocol: if the job is relaunched
        first, the restart replays the whole protocol from a clean state."""
        try:
            marker_mtime = os.path.getmtime(marker)
        except OSError:
            marker_mtime = time.time()
        logger.info(
            "Urgent-save: observing %s to flip past iteration %s (marker set at %s)",
            tracker_path(save_dir),
            base_iteration,
            marker_mtime,
        )
        while True:
            time.sleep(self._poll_interval)
            iteration, tracker_mtime = read_tracker(save_dir)
            flipped = iteration is not None and (
                base_iteration is None or iteration > base_iteration
            )
            if flipped and tracker_mtime > marker_mtime:
                self._publish(
                    cmd_ts,
                    STATUS_SAVED,
                    "tracker flipped to iteration {}".format(iteration),
                )
                logger.info(
                    "Urgent-save: checkpoint completed, tracker=%s (request %s).",
                    iteration,
                    cmd_ts,
                )
                return

    def _publish(self, cmd_ts, status, message=""):
        try:
            self._client.kv_store_set(
                result_key(cmd_ts), build_result(cmd_ts, status, message)
            )
        except Exception as e:  # noqa: BLE001 - never break the listener
            logger.error(
                "Urgent-save: fail to publish '%s' result: %s", status, e
            )
