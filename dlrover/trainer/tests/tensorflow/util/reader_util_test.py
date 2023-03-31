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
from dlrover.trainer.tensorflow.util.reader_util import reader_registery


class ReaderRegister(unittest.TestCase):
    def test_register_reader(self):
        class TestReader:
            def __init__(self):
                self.a = 1
        reader_registery.register_reader("test_reader", TestReader)
        Reader = reader_registery.get_reader("test_reader")
        reader = Reader()
        self.assertEqual(reader.a, 1)

    def test_unregister_reader(self):
        reader_registery.unregister_reader("test_reader")
        Reader = reader_registery.get_reader("test_reader")
        self.assertEqual(Reader, None)
 


if __name__ == "__main__":
    unittest.main(verbosity=2)
