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

import asyncio
from concurrent.futures import Future
from threading import Thread
from typing import Any, Coroutine, TypeVar

"""Utility functions for asynchronous operations in DLRover.

Provides functions to bridge between synchronous and asynchronous code,
allowing for running coroutines in a thread-safe manner and managing the main event loop.
"""


T = TypeVar("T")

__main_loop = None


def init_main_loop():
    global __main_loop
    try:
        __main_loop = asyncio.get_running_loop()
    except RuntimeError:
        raise RuntimeError(
            "You must call init_main_loop() in the main thread or ray AsyncActor"
        )


def unsafe_run_blocking(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine in a separate thread and wait for its result.

    You should check before use this function:
    1. It could not use main loop. In general, all coroutines should be in the main loop.
    2. It must sync call.
    3. It's safe to block current event loop.
    4. The coroutine does not spawn background tasks.
    """
    assert asyncio.get_event_loop() == __main_loop, (
        "unsafe_run_blocking is prepared for main loop"
    )

    result = Future[T]()

    def run():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            task = loop.create_task(coro)
            loop.run_until_complete(task)

            # Check for unfinished tasks to avoid resource leaks
            leftovers = [t for t in asyncio.all_tasks(loop) if not t.done()]
            assert len(leftovers) == 0, (
                f"Should not spawn background tasks in unsafe_run_blocking: {leftovers}"
            )
            result.set_result(task.result())
        except Exception as e:
            result.set_exception(e)

    Thread(target=run).start()
    return result.result()


def is_in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def get_main_loop() -> asyncio.AbstractEventLoop:
    """Get the main event loop."""
    global __main_loop
    if __main_loop is None:
        raise RuntimeError(
            "Main loop is not initialized. Call init_main_loop first."
        )
    return __main_loop


def as_future(co: Coroutine[Any, Any, T]) -> Future[T]:
    """Run a coroutine in the main loop and return a Future."""
    loop = get_main_loop()
    assert not is_in_event_loop(), "Should not be called in an event loop"
    return asyncio.run_coroutine_threadsafe(co, loop)


def completed_future(result: T) -> Future[T]:
    """Create a completed Future with the given result."""
    future = Future[T]()
    future.set_result(result)
    return future


def wait(co: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine in the main loop and wait for its result."""
    return as_future(co).result()
