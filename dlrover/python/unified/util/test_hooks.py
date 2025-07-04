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

import os


def init_coverage():
    if "COVERAGE_PROCESS_START" not in os.environ:
        return
    try:
        import coverage

        coverage.process_startup()
    except ImportError:
        pass


def coverage_enabled():
    """Check if coverage is enabled."""
    try:
        import coverage

        return coverage.Coverage.current() is not None
    except ImportError:
        return False
