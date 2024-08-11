# Copyright 2022 The DLRover Authors. All rights reserved.
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

import dlrover.python.util.file_util as fu


class FileUtilTest(unittest.TestCase):
    def test_is_same_path(self):
        self.assertTrue(fu.is_same_path("/foo/bar", "/foo/bar"))
        self.assertTrue(fu.is_same_path("/foo/bar", "/foo//bar"))
        self.assertFalse(fu.is_same_path("/foo/bar", "/foo/bar0"))

    def test_find_file_in_parents(self):
        self.assertIsNotNone(fu.find_file_in_parents("setup.py"))
