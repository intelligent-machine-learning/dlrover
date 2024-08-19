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
import shutil
import tempfile
import time
import unittest

from dlrover.python.common.storage import (
    KeepLatestStepStrategy,
    KeepStepIntervalStrategy,
    PosixDiskStorage,
    get_checkpoint_storage,
)


class TestLocalStrategyGenerator(unittest.TestCase):
    def test_keep_latest_deletion_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deletion_strategy = KeepLatestStepStrategy(3, tmpdir)
            os.makedirs(os.path.join(tmpdir, "1"))
            for step in range(2, 5):
                os.makedirs(os.path.join(tmpdir, str(step)))
                deletion_strategy.clean_up(step - 1, shutil.rmtree)
            self.assertListEqual(sorted(os.listdir(tmpdir)), ["2", "3", "4"])

    def test_keep_step_interval_deletion_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deletion_strategy = KeepStepIntervalStrategy(4, tmpdir)
            os.makedirs(os.path.join(tmpdir, "2"))
            for i in range(2, 6):
                step = i * 2
                os.makedirs(os.path.join(tmpdir, str(step)))
                deletion_strategy.clean_up(step - 2, shutil.rmtree)
            self.assertListEqual(sorted(os.listdir(tmpdir)), ["10", "4", "8"])

    def test_posix_storage_with_deletion(self):
        tracker_file = "dlrover_latest.txt"

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker_file_path = os.path.join(tmpdir, tracker_file)
            strategy = KeepLatestStepStrategy(
                max_to_keep=3, checkpoint_dir=tmpdir
            )
            storage = get_checkpoint_storage(strategy)

            for step in range(1, 5):
                storage.write(str(step), tracker_file_path)
                os.makedirs(os.path.join(tmpdir, str(step)))
                storage.commit(step, True)

            files = os.listdir(tmpdir)
            files.remove(tracker_file)
            self.assertListEqual(sorted(files), ["2", "3", "4"])

            class_meta = storage.get_class_meta()
            self.assertEqual(class_meta.class_name, "PosixStorageWithDeletion")

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker_file_path = os.path.join(tmpdir, tracker_file)
            strategy = KeepStepIntervalStrategy(
                keep_interval=4, checkpoint_dir=tmpdir
            )
            storage = get_checkpoint_storage(strategy)
            for step in range(1, 9):
                storage.write(str(step), tracker_file_path)
                os.makedirs(os.path.join(tmpdir, str(step)))
                storage.commit(step, True)

            files = os.listdir(tmpdir)
            files.remove(tracker_file)
            self.assertListEqual(sorted(files), ["4", "8"])

            storage = get_checkpoint_storage(None)
            self.assertTrue(isinstance(storage, PosixDiskStorage))

    def test_error_write(self):
        storage = PosixDiskStorage()
        path = "/tmp/" + str(time.time()) + "/test_error_write"
        try:
            storage.write("testing123", path)
            self.fail()
        except OSError:
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
