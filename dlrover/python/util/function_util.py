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

import functools
import signal
import time
import traceback
from concurrent import futures

from dlrover.python.common.log import default_logger as logger

TIMEOUT_MAX = 24 * 60 * 60


class TimeoutException(Exception):
    pass


def timeout(secs=-1, callback_func=None):
    """Decorator for timeout controlled function using."""

    if callback_func is None:
        if secs <= 0:
            timeout_secs_value = TIMEOUT_MAX
        else:
            timeout_secs_value = secs
    else:
        timeout_secs_value = callback_func()

    def decorator(func):
        def handler(signum, frame):
            raise TimeoutException("Function call timed out")

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout_secs_value)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapped

    return decorator


def threading_timeout(secs=-1, callback_func=None):
    """
    Decorator for timeout that limits the execution
    time of functions executed in main and non-main threads
    :param secs: timeout seconds
    :param callback_func: the function that set the timeout
    """
    if callback_func is None:
        if secs <= 0:
            timeout_secs_value = TIMEOUT_MAX
        else:
            timeout_secs_value = secs
    else:
        timeout_secs_value = callback_func()

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout_secs_value)
                except futures.TimeoutError:
                    raise TimeoutException("Function call timed out")
                return result

        return wrapped

    return decorator


def retry(retry_times=10, retry_interval=5, raise_exception=True):
    """Decorator for function with retry mechanism using."""

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            exception = None
            for i in range(retry_times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    class_name = func.__class__.__name__
                    func_name = func.__name__
                    tb = traceback.format_exception(
                        type(e), e, e.__traceback__, limit=3
                    )
                    logger.warning(
                        f"Retry {i} to {class_name}.{func_name} with failure {e}"
                    )
                    logger.debug(f"Caused traceback: {tb}")
                    exception = e
                    time.sleep(retry_interval)
            if exception:
                logger.error(exception)
                if raise_exception:
                    raise exception
            return None

        return wrapped

    return decorator


def ignore_exceptions():
    """Decorator for function with exception ignoring using."""

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                class_name = func.__class__.__name__
                func_name = func.__name__
                tb = traceback.format_exception(
                    type(e), e, e.__traceback__, limit=3
                )
                logger.warning(
                    f"Invocation with {class_name}.{func_name} with failure {e}, ",
                    f"with traceback {tb}",
                )
            return None

        return wrapped

    return decorator
