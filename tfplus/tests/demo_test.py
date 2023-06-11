# Copyright 2023 tfplus.

import unittest

from tfplus.python.demo import print_localtime


class DemoTest(unittest.TestCase):
    def test_demo(self):
        # TODO: fix print_localtime export issue.
        print_localtime()


if __name__ == "__main__":
    unittest.main()
