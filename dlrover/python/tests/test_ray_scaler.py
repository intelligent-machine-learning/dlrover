# Copyright 2022 The EasyDL Authors. All rights reserved.
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
import unittest

from dlrover.python.master.watcher.ray_watcher import ActorWatcher


class ActorWatcherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.system("ray stop --force")
        r = os.system("ray start --head --port=5001  --dashboard-port=5000")
        assert r == 0
        cls.actor_watcher = ActorWatcher("test", "")
