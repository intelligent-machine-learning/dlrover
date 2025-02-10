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

from dlrover.python.master.args import (
    parse_master_args,
    parse_tuple2_list,
    str2bool,
)


class ArgsTest(unittest.TestCase):
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

    def test_parse_tuple2_list(self):
        self.assertEqual(parse_tuple2_list(""), [])
        self.assertEqual(parse_tuple2_list("[]"), [])
        self.assertEqual(parse_tuple2_list("[('1', '2')]"), [("1", "2")])
        self.assertEqual(
            parse_tuple2_list("[('1', '2'), ('3', '4')]"),
            [("1", "2"), ("3", "4")],
        )

    def test_parse_master_args(self):
        original_args = [
            "--job_name",
            "test",
            "--namespace",
            "default",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.job_name, "test")
        self.assertTrue(parsed_args.namespace, "default")
        self.assertEqual(parsed_args.pending_timeout, 900)
        self.assertEqual(parsed_args.pending_fail_strategy, 1)
        self.assertTrue(parsed_args.service_type, "grpc")

        original_args = [
            "--job_name",
            "test",
            "--namespace",
            "default",
            "--pending_timeout",
            "600",
            "--pending_fail_strategy",
            "2",
            "--service_type",
            "http",
            "--pre_check_ops",
            "[('t1', 't2')]",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.pending_timeout, 600)
        self.assertEqual(parsed_args.pending_fail_strategy, 2)
        self.assertTrue(parsed_args.service_type, "http")
        self.assertEqual(parsed_args.pre_check_ops, [("t1", "t2")])

        original_args = [
            "--job_name",
            "test",
            "--hang_detection",
            "1",
            "--hang_downtime",
            "15",
            "--xpu_type",
            "ascend",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.job_name, "test")
        self.assertEqual(parsed_args.hang_detection, 1)
        self.assertEqual(parsed_args.hang_downtime, 15)
        self.assertEqual(parsed_args.xpu_type, "ascend")

        original_args = [
            "--job_name",
            "test",
            "--xpu_type",
            "nvidia",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.xpu_type, "nvidia")
