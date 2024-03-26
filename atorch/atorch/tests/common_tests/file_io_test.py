import os
import shutil
import tempfile
import unittest
from os.path import join as path_join

import atorch


class DataSetTest(unittest.TestCase):
    def setUp(self):
        atorch.file_io
        self._local_tmp_dir = tempfile.mktemp()
        super(DataSetTest, self).setUp()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir)
        super(DataSetTest, self).tearDown()

    def _file_io_test(self, uri):
        os.mkdir(path_join(uri, "path1"))
        os.makedirs(path_join(uri, "path2/path1"))
        self.assertEqual(os.listdir(uri), ["path1", "path2"])
        self.assertEqual(path_join(uri, "path2"), ["path1"])

        value = "atorch write test."
        with open(path_join(uri, "out.txt"), "w") as fd:
            fd.write(value)
        with open(path_join(uri, "out.txt"), "r") as fd:
            content = fd.read()
        self.assertEqual(content, value)

        os.rename(path_join(uri, "out.txt"), path_join(uri, "in.txt"))
        self.assertEqual(os.listdir(uri), ["path1", "path2", "in.txt"])
        with open(path_join(uri, "in.txt"), "r") as fd:
            content = fd.read()
        self.assertEqual(content, value)

        os.remove(path_join(uri, "in.txt"))
        self.assertEqual(os.listdir(uri), ["path1", "path2"])

    def local_file_io_test(self):
        self._file_io_test(self.local_file_io_test)


if __name__ == "__main__":
    unittest.main()
