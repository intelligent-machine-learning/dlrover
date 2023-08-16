# Copyright 2023 The DLRover Authors. All rights reserved.
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

from dlrover.trainer.torch.elastic_run import (
    _check_dlrover_master_available,
    _launch_dlrover_local_master,
)


class ElasticRunTest(unittest.TestCase):
    def test_launch_local_master(self):
        handler, addr = _launch_dlrover_local_master()
        available = _check_dlrover_master_available(addr)
        self.assertTrue(available)
        handler.close()
