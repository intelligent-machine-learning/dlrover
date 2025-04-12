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


def wait_all(store, rank: int, prefix: str, size: int):
    keys = []
    for idx in range(size):
        key = f"{prefix}{idx}"
        keys.append(key)
    store.wait(keys)
    store.set(f"{prefix}{rank}.FIN", b"FIN")
    if rank == 0:
        # Rank0 runs the TCPStore daemon, as a result it needs to exit last.
        # Otherwise, the barrier may timeout if rank0 process finished the work
        # before other processes finished `wait_all` method
        for node_rank in range(size):
            store.get(f"{prefix}{node_rank}.FIN")


def synchronize(
    store,
    data: bytes,
    rank: int,
    world_size: int,
    key_prefix: str,
    barrier_timeout: float = 300,
):
    store.set_timeout(timedelta(seconds=barrier_timeout))
    store.set(f"{key_prefix}{rank}", data)
    wait_all(store, rank, key_prefix, world_size)


def barrier(
    store,
    rank: int,
    world_size: int,
    key_prefix: str,
    barrier_timeout: float = 300,
) -> None:
    data = f"{rank}".encode()
    synchronize(store, data, rank, world_size, key_prefix, barrier_timeout)
