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

import time
from contextlib import contextmanager

from dlrover.python.common.log import default_logger as logger


@contextmanager
def catch_exception(msg: str):
    """Catch exception and log it."""
    try:
        yield
    except Exception:
        logger.exception(msg)


@contextmanager
def log_execution(name: str, log_exception: bool = True):
    """Log the execution of a block of code."""
    logger.info(f"Run '{name}' ...")
    start = time.time()
    try:
        yield
        elapsed = time.time() - start
        logger.info(f"End '{name}' successfully, took {elapsed:.2f} seconds.")
    except Exception:
        if log_exception:
            logger.exception(f"Error during execution of '{name}'")
        raise
