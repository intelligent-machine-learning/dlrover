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

from pytest import fixture

from dlrover.python.unified.util.test_hooks import _RESET_HOOKS

"""Fixture for testing purposes, not associated with any specific component."""


@fixture(autouse=True)
def reset_all_singletons():
    """Reset all singleton instances."""
    yield
    for reset_hook in _RESET_HOOKS:
        reset_hook()
