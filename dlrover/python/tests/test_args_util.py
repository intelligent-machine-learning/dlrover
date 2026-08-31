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

import argparse
import unittest

from dlrover.python.util.args_util import (
    parse_group_affinity,
    parse_tuple_dict,
    parse_tuple_list,
    str2bool,
    validate_group_affinity_equal_size,
)


class ArgsUtilTest(unittest.TestCase):
    def test_str2bool(self):
        self.assertTrue(str2bool("TRUE"))
        self.assertTrue(str2bool("True"))
        self.assertTrue(str2bool("true"))
        self.assertTrue(str2bool("yes"))
        self.assertTrue(str2bool("t"))
        self.assertTrue(str2bool("y"))
        self.assertTrue(str2bool("1"))
        self.assertTrue(str2bool(True))

        self.assertFalse(str2bool("FALSE"))
        self.assertFalse(str2bool("False"))
        self.assertFalse(str2bool("false"))
        self.assertFalse(str2bool("no"))
        self.assertFalse(str2bool("n"))
        self.assertFalse(str2bool("0"))
        self.assertFalse(str2bool(False))

    def test_parse_tuple_list(self):
        self.assertEqual(parse_tuple_list(""), [])
        self.assertEqual(parse_tuple_list("[]"), [])
        self.assertEqual(parse_tuple_list("[('1', '2')]"), [("1", "2")])
        self.assertEqual(
            parse_tuple_list("[('1', '2'), ('3', '4', '5')]"),
            [("1", "2"), ("3", "4", "5")],
        )
        self.assertEqual(
            parse_tuple_list("[('1', '2'), ('3', '4', True)]"),
            [("1", "2"), ("3", "4", True)],
        )

    def test_parse_tuple_dict(self):
        # valid
        self.assertEqual(parse_tuple_dict(""), {})
        self.assertEqual(parse_tuple_dict("{}"), {})
        self.assertEqual(
            parse_tuple_dict("{('1', '2'):True}"), {("1", "2"): True}
        )
        self.assertEqual(
            parse_tuple_dict("{('1', '2'):'true'}"), {("1", "2"): True}
        )
        self.assertEqual(
            parse_tuple_dict("{('1', '2'):'t', ('3', '4'):'no'}"),
            {("1", "2"): True, ("3", "4"): False},
        )

        # invalid
        try:
            parse_tuple_dict("{('1', '2'):true}")
            self.fail()
        except Exception:
            pass
        try:
            parse_tuple_dict("{'1':True}")
            self.fail()
        except Exception:
            pass

    def test_parse_group_affinity(self):
        # empty / None -> None
        self.assertIsNone(parse_group_affinity(None))
        self.assertIsNone(parse_group_affinity(""))

        # valid
        self.assertEqual(
            parse_group_affinity("{0: 10, 1: 15}"), {0: 10, 1: 15}
        )
        self.assertEqual(parse_group_affinity("{2: 3}"), {2: 3})
        # keys are ordered by value usage, not by dict order
        self.assertEqual(
            parse_group_affinity("{1: 15, 0: 10}"), {1: 15, 0: 10}
        )

        # invalid
        for bad in ["[1, 2]", "abc", "[]", "'not a dict'", "{0: 0}", "{}"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_group_affinity(bad)
        # negative group id
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_group_affinity("{-1: 5}")
        # non-int value
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_group_affinity("{0: 'x'}")
        # bool value (True is an int subclass, must be rejected)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_group_affinity("{0: True}")

    def test_validate_group_affinity_equal_size(self):
        # None / empty -> None (not applicable)
        self.assertIsNone(validate_group_affinity_equal_size(None))
        self.assertIsNone(validate_group_affinity_equal_size({}))

        # equal sizes -> the single size
        self.assertEqual(validate_group_affinity_equal_size({0: 8, 1: 8}), 8)
        self.assertEqual(
            validate_group_affinity_equal_size({2: 128, 0: 128, 1: 128}), 128
        )

        # unequal sizes -> ValueError
        with self.assertRaisesRegex(ValueError, "equal"):
            validate_group_affinity_equal_size({0: 10, 1: 15})
