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

from dlrover.python.unified.master.state_backend import (
    MasterStateBackendFactory,
    RayInternalMasterStateBackend,
)
from dlrover.python.unified.tests.base import RayBaseTest
from dlrover.python.unified.tests.master.base import BaseMasterTest


class StateBackendTest(BaseMasterTest):
    def test_state_backend_factory(self):
        self.assertTrue(
            isinstance(
                MasterStateBackendFactory.get_state_backend(),
                RayInternalMasterStateBackend,
            )
        )


class RayInternalMasterStateBackendTest(RayBaseTest):
    def setUp(self):
        super().setUp()
        self._key_prefix = "ut_test_"
        self._backend = RayInternalMasterStateBackend()
        self.init_ray_safely()

    def tearDown(self):
        self._backend.reset(self._key_prefix)
        self.close_ray_safely()
        super().tearDown()

    def test_basic(self):
        test_key = self._key_prefix + "k1"
        test_key_b = test_key.encode()
        self._backend.set(test_key_b, b"v1")
        self.assertEqual(self._backend.get(test_key_b), b"v1")
        self.assertTrue(self._backend.exists(test_key_b))
        self._backend.delete(test_key_b)
        self.assertFalse(self._backend.exists(test_key_b))
