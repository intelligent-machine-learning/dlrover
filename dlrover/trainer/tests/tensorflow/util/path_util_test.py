import os
import unittest

from dlrover.trainer.tensorflow.util.path_util import parse_uri
from dlrover.trainer.constants.tf_constants import TFConstants

class BaseExecutorTest(unittest.TestCase):
    def test_parse_file_scheme(self):
        import pdb 
        pdb.set_trace()
        path = "file:///home/test/data.csv"
        scheme, file_path = parse_uri(path)
        self.assertEqual(scheme, TFConstants.FILE_SCHEME())
        self.assertEqual(file_path, "/home/test/data.csv")
        path = "file://./home/test/data.csv"
        scheme, file_path  = parse_uri(path)
        self.assertEqual(scheme, TFConstants.FILE_SCHEME())
        self.assertEqual(file_path, "./home/test/data.csv")

    def test_parse_fake_scheme(self):
        path = "fake:///home/test/data.csv"
        scheme, _  = parse_uri(path)
        self.assertEqual(scheme, TFConstants.FAKE_SCHEME())
 
if __name__ == "__main__":
    unittest.main(verbosity=2)
