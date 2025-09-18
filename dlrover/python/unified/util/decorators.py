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

import inspect
import time
from functools import wraps

from dlrover.python.common.log import default_logger as logger


def catch_exception(msg: str):
    """Catch exception and log it."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                logger.exception(msg, stacklevel=3)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception:
                logger.exception(msg, stacklevel=3)

        return async_wrapper if inspect.iscoroutinefunction(func) else wrapper

    return decorator


class _LogExecution:
    def __init__(self, name: str, log_exception: bool = True):
        self.name = name
        self.log_exception = log_exception
        self.stacklevel = 2

    def __enter__(self):
        logger.info(f"Run '{self.name}' ...", stacklevel=self.stacklevel)
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            logger.info(
                f"End '{self.name}' successfully, took {elapsed:.2f} seconds.",
                stacklevel=self.stacklevel,
            )
        else:
            if self.log_exception:
                logger.exception(
                    f"Error during execution of '{self.name}'",
                    stacklevel=self.stacklevel,
                )
            # Do not suppress the exception
            return False

    # as decorator
    def __call__(self, func):
        self.stacklevel += 1

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with self:
                return await func(*args, **kwargs)

        return async_wrapper if inspect.iscoroutinefunction(func) else wrapper


def log_execution(name: str, log_exception: bool = True):
    """
    Log the execution of a block of code.

    Note: Decorating both base and overridden functions may result in nested logging.
    """
    return _LogExecution(name, log_exception=log_exception)
