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
from pathlib import Path
from tempfile import TemporaryDirectory

import dlrover.python.util.file_util as fu


class FileUtilTest(unittest.TestCase):
    def test_is_same_path(self):
        self.assertTrue(fu.is_same_path("/foo/bar", "/foo/bar"))
        self.assertTrue(fu.is_same_path("/foo/bar", "/foo//bar"))
        self.assertFalse(fu.is_same_path("/foo/bar", "/foo/bar0"))

    def test_find_file_in_parents(self):
        self.assertIsNotNone(fu.find_file_in_parents("setup.py"))

    def test_read_last_n_lines_across_multiple_blocks(self):
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "training.log"
            with file_path.open("w", encoding="utf-8") as file:
                for index in range(1000):
                    file.write(f"line-{index:04d}-{'x' * 100}\n")

            all_lines = fu.read_last_n_lines(str(file_path), 2000)
            last_ten_lines = fu.read_last_n_lines(str(file_path), 10)
            no_lines = fu.read_last_n_lines(str(file_path), 0)

        self.assertEqual(len(all_lines), 1000)
        self.assertTrue(all_lines[0].startswith(b"line-0000-"))
        self.assertTrue(all_lines[-1].startswith(b"line-0999-"))
        self.assertEqual(len(last_ten_lines), 10)
        self.assertTrue(last_ten_lines[0].startswith(b"line-0990-"))
        self.assertTrue(last_ten_lines[-1].startswith(b"line-0999-"))
        self.assertEqual(no_lines, [])
