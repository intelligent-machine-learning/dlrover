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
import unittest

import ray
from ray.experimental.internal_kv import (
    _initialize_internal_kv,
    _internal_kv_initialized,
)

from dlrover.python.rl.master.state_backend import (
    MasterStateBackendFactory,
    RayInternalMasterStateBackend,
)
from dlrover.python.rl.tests.master.base import BaseMasterTest


class StateBackendTest(BaseMasterTest):
    def test_state_backend_factory(self):
        self.assertTrue(
            isinstance(
                MasterStateBackendFactory.get_state_backend(),
                RayInternalMasterStateBackend,
            )
        )


class RayInternalMasterStateBackendTest(unittest.TestCase):
    def setUp(self):
        self._key_prefix = "ut_test_"
        self._backend = RayInternalMasterStateBackend()
        if not ray.is_initialized():
            ray.init(num_cpus=1, include_dashboard=False)

    def tearDown(self):
        self._backend.reset(self._key_prefix)
        ray.shutdown()

    def test_basic(self):
        test_key = self._key_prefix + "k1"
        self._backend.set(test_key.encode(), b"v1")
        self.assertEqual(self._backend.get(test_key.encode()), b"v1")
        self.assertTrue(self._backend.exists(test_key.encode()))
        self._backend.delete(test_key.encode())
        self.assertFalse(self._backend.exists(test_key.encode()))
