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

from datetime import timedelta

from dlrover.python.common.log import default_logger as logger

_NUM_MEMBERS = "/num_members"
_LAST_MEMBER_CHECKIN = "/last_member"


def _barrier_nonblocking(store, world_size: int, key_prefix: str) -> str:
    """
    Does all the non-blocking operations for a barrier and returns
    the final key that can be waited on.
    """
    num_members_key = key_prefix + _NUM_MEMBERS
    last_member_key = key_prefix + _LAST_MEMBER_CHECKIN

    idx = store.add(num_members_key, 1)
    logger.info(f"Add {num_members_key}: {idx} {world_size}")
    if idx == world_size:
        logger.info(f"Set {last_member_key}: {idx} {world_size}")
        store.set(last_member_key, "<val_ignored>")

    return last_member_key


def barrier(
    store, world_size: int, key_prefix: str, barrier_timeout: float = 300
) -> None:
    """
    A global lock between agents. This will pause all workers until at least
    ``world_size`` workers respond.

    This uses a fast incrementing index to assign waiting ranks and a success
    flag set by the last worker.

    Time complexity: O(1) per worker, O(N) globally.

    Note: Since the data is not removed from the store, the barrier can be used
        once per unique ``key_prefix``.
    """

    store.set_timeout(timedelta(seconds=barrier_timeout))
    last_member_key = _barrier_nonblocking(
        store=store, world_size=world_size, key_prefix=key_prefix
    )
    value = store.get(last_member_key)
    logger.info(f"Get {last_member_key}: {value}")
