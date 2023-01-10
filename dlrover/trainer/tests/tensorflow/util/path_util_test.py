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

from dlrover.trainer.constants.tf_constants import TFConstants
from dlrover.trainer.tensorflow.util.path_util import parse_uri


class BaseExecutorTest(unittest.TestCase):
    def test_parse_file_scheme(self):
        path = "file:///home/test/data.csv"
        scheme, file_path = parse_uri(path)
        self.assertEqual(scheme, TFConstants.FILE_SCHEME())
        self.assertEqual(file_path, "/home/test/data.csv")
        path = "file://./home/test/data.csv"
        scheme, file_path = parse_uri(path)
        self.assertEqual(scheme, TFConstants.FILE_SCHEME())
        self.assertEqual(file_path, "./home/test/data.csv")

    def test_parse_fake_scheme(self):
        path = "fake:///home/test/data.csv"
        scheme, _ = parse_uri(path)
        self.assertEqual(scheme, TFConstants.FAKE_SCHEME())


if __name__ == "__main__":
    unittest.main(verbosity=2)
