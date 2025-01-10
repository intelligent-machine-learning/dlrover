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
from unittest.mock import patch

from dlrover.trainer.torch.utils import (
    version_less_than_230,
    version_less_than_240,
)


class TestTorchVersionFunctions(unittest.TestCase):
    def test_version_less_than_230(self):
        with patch("torch.__version__", "2.2.1"):
            self.assertTrue(
                version_less_than_230(), "Expected True for version < 2.2.2"
            )

        with patch("torch.__version__", "2.2.1+cu111"):
            self.assertTrue(
                version_less_than_230(), "Expected True for version < 2.2.2"
            )

        with patch("torch.__version__", "2.2.2+cu111"):
            self.assertTrue(
                version_less_than_230(), "Expected True for version = 2.2.2"
            )

        with patch("torch.__version__", "2.3.0+cu111"):
            self.assertFalse(
                version_less_than_230(), "Expected False for version > 2.2.2"
            )

    def test_version_less_than_240(self):
        with patch("torch.__version__", "2.3.0"):
            self.assertTrue(
                version_less_than_240(), "Expected True for version < 2.3.1"
            )

        with patch("torch.__version__", "2.3.0+cu111"):
            self.assertTrue(
                version_less_than_240(), "Expected True for version < 2.3.1"
            )

        with patch("torch.__version__", "2.3.1+cu111"):
            self.assertTrue(
                version_less_than_240(), "Expected True for version = 2.3.1"
            )

        with patch("torch.__version__", "2.4.0+cu111"):
            self.assertFalse(
                version_less_than_240(), "Expected False for version > 2.3.1"
            )
