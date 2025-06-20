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

import torch

from dlrover.python.unified.common.transfer import DataTransfer
from dlrover.python.unified.tests.base import BaseTest


class DataTransferTest(BaseTest):
    def test_basic(self):
        data = DataTransfer(
            {
                "tensor_a": torch.zeros(3, 3),
                "tensor_b": torch.ones(2, 2),
            },
            {"k1": "v1"},
        )

        self.assertIsNotNone(data.tensor)
        self.assertEqual(data.user_data, {"k1": "v1"})
        self.assertEqual(data.get_data("k1"), "v1")
