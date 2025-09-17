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

from dlrover.python.unified.util.decorators import (
    catch_exception,
    log_execution,
)


async def test_catch_exception():
    @catch_exception("test_sync")
    def test_sync(x):
        raise ValueError("test")

    @catch_exception("test_async")
    async def test_async(x):
        raise ValueError("test")

    assert test_sync(3) is None  # caught
    assert await test_async(3) is None  # caught


async def test_log_execution():
    with log_execution("test_block"):
        pass

    @log_execution("test_func")
    def test_func():
        pass

    @log_execution("test_async_func")
    async def test_async_func():
        await asyncio.sleep(0.1)

    test_func()
    await test_async_func()
