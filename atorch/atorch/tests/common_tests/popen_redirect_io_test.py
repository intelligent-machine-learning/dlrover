# coding=utf-8
from __future__ import absolute_import, annotations, unicode_literals

import tempfile
import unittest
from subprocess import Popen


class TestRedirectIO(unittest.TestCase):
    def testIO(self):
        """Write stdout/stderr to fd and check result"""
        with tempfile.NamedTemporaryFile() as stdout, tempfile.NamedTemporaryFile() as stderr:
            cmd = [
                "python",
                "-c",
                "import sys;print(1, file=sys.stderr, flush=True);print(2, file=sys.stdout, flush=True)",
            ]
            process = Popen(cmd, shell=False, stdout=stdout, stderr=stderr, universal_newlines=True)
            process.wait()
            stdout.seek(0)
            stderr.seek(0)
            self.assertEqual(b"1\n", stderr.readline())
            self.assertEqual(b"2\n", stdout.readline())


if __name__ == "__main__":
    unittest.main()
