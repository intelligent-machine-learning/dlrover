#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

from dlrover.python.common.global_context import Context


class SingletonTest(unittest.TestCase):
    def test_singleton(self):
        context0 = Context.singleton_instance()
        context1 = Context.singleton_instance()
        context2 = Context.singleton_instance()

        self.assertTrue(id(context0) == id(context1) == id(context2))
