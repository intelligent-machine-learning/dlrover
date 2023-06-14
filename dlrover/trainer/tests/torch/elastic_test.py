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

import os
import unittest

from dlrover.trainer.torch.elastic import set_master_addr


class ElasticSetupTest(unittest.TestCase):
    def test_set_master_addr(self):
        os.environ["RANK"] = "0"
        os.environ["MASTER_PORT"] = "1234"
        os.environ["DLROVER_MASTER_ADDR"] = "localhost:1234"
        set_master_addr()
        self.assertTrue(os.environ["MASTER_ADDR"] != "")
        self.assertTrue(os.environ["MASTER_PORT"] != "")
