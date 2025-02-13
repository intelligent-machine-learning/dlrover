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

TIMEOUT_MAX = 24 * 60 * 60


class TimeoutException(Exception):
    pass


def timeout(secs=-1, callback_func=None):
    """Decorator for timeout controlled function using."""

    if callback_func is None:
        timeout_secs_value = secs
    else:
        timeout_secs_value = callback_func()

    if timeout_secs_value < 0:
        timeout_secs_value = TIMEOUT_MAX

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
