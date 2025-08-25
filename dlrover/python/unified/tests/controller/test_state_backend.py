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

from dlrover.python.unified.controller.state_backend import (
    RayInternalMasterStateBackend,
)


def test_basic(shared_ray):
    key_prefix = "ut_test_"
    backend = RayInternalMasterStateBackend()

    test_key = key_prefix + "k1"
    test_key_b = test_key.encode()
    backend.set(test_key_b, b"v1")
    assert backend.get(test_key_b) == b"v1"
    assert backend.exists(test_key_b)
    backend.delete(test_key_b)
    assert not backend.exists(test_key_b)
    backend.reset(key_prefix)
