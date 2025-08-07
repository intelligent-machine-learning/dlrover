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
from typing import Any, Coroutine, TypeVar

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


def wait(co: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine in the main loop and wait for its result."""
    return as_future(co).result()
