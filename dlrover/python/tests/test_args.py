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

from dlrover.python.master.args import parse_master_args
from dlrover.python.util.common_util import print_args


class ArgsTest(unittest.TestCase):
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
        self.assertTrue(parsed_args.pre_check_ops)
        self.assertTrue(parsed_args.task_process_timeout, 1800)

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
            "[('t1', 't2', 'y')]",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.pending_timeout, 600)
        self.assertEqual(parsed_args.pending_fail_strategy, 2)
        self.assertTrue(parsed_args.service_type, "http")
        self.assertEqual(parsed_args.pre_check_ops, [("t1", "t2", "y")])

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

        original_args = [
            "--job_name",
            "test",
            "--xpu_type",
            "mthreads",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.xpu_type, "mthreads")

        # test print
        print_args(parsed_args, groups=[["optimizer", "loss"]])

        # test invalid
        original_args = [
            "--job_name",
            "test",
            "--xpu_type",
            "nvidia",
            "--hang_downtime",
            "-1",
        ]
        with self.assertRaises(SystemExit) as cm:
            parse_master_args(original_args)
            self.assertEqual(cm.exception.code, 2)

    def test_parse_group_affinity(self):
        # default is None when the argument is absent
        original_args = [
            "--job_name",
            "test",
            "--namespace",
            "default",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertIsNone(parsed_args.group_affinity)

        # valid mapping with spaces inside the literal dict
        original_args = [
            "--job_name",
            "test",
            "--namespace",
            "default",
            "--group-affinity={0: 10, 1: 15}",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.group_affinity, {0: 10, 1: 15})

        # the underscore alias is also accepted
        original_args = [
            "--job_name",
            "test",
            "--group_affinity={2: 3}",
        ]
        parsed_args = parse_master_args(original_args)
        self.assertEqual(parsed_args.group_affinity, {2: 3})

    def test_parse_soft_group_args(self):
        # defaults: no soft mapping, the failover flag off
        parsed_args = parse_master_args(["--job_name", "test"])
        self.assertIsNone(parsed_args.soft_group_affinity)
        self.assertFalse(parsed_args.no_group_failover)

        # soft group affinity reuses the dict parser (unequal sizes)
        parsed_args = parse_master_args(
            [
                "--job_name",
                "test",
                "--soft-group-affinity={0: 30, 1: 20}",
                "--no-group-failover",
            ]
        )
        self.assertEqual(parsed_args.soft_group_affinity, {0: 30, 1: 20})
        self.assertTrue(parsed_args.no_group_failover)

        # the underscore aliases are also accepted
        parsed_args = parse_master_args(
            [
                "--job_name",
                "test",
                "--soft_group_affinity={2: 3}",
                "--no_group_failover",
            ]
        )
        self.assertEqual(parsed_args.soft_group_affinity, {2: 3})
        self.assertTrue(parsed_args.no_group_failover)

        # invalid mapping makes the parser exit with code 2
        original_args = [
            "--job_name",
            "test",
            "--group-affinity={0: 0}",
        ]
        with self.assertRaises(SystemExit) as cm:
            parse_master_args(original_args)
        self.assertEqual(cm.exception.code, 2)

    def test_parse_topology_rerank_args(self):
        # default off
        parsed = parse_master_args(["--job_name", "test"])
        self.assertFalse(parsed.enable_topology_rerank)

        # the dash and underscore aliases both enable the rerank
        parsed = parse_master_args(
            ["--job_name", "test", "--enable-topology-rerank"]
        )
        self.assertTrue(parsed.enable_topology_rerank)
        parsed = parse_master_args(
            ["--job_name", "test", "--enable_topology_rerank"]
        )
        self.assertTrue(parsed.enable_topology_rerank)

    def test_parse_node_group_strategy(self):
        # defaults: no strategy + all parallel sizes at their defaults
        # (etp stays None, i.e. it inherits tp at the master layer)
        parsed = parse_master_args(["--job_name", "test"])
        self.assertIsNone(parsed.node_group_strategy)
        self.assertEqual(parsed.tensor_model_parallel_size, 1)
        self.assertEqual(parsed.pipeline_model_parallel_size, 1)
        self.assertEqual(parsed.expert_model_parallel_size, 1)
        self.assertIsNone(parsed.expert_tensor_parallel_size)
        self.assertEqual(parsed.context_parallel_size, 1)

        # ep_pp_dp with parallel sizes (long and short aliases)
        parsed = parse_master_args(
            [
                "--job_name",
                "test",
                "--node-group-strategy=ep_pp_dp",
                "--group-affinity={0: 128, 1: 128}",
                "--tp",
                "1",
                "--pp",
                "16",
                "--ep",
                "32",
                "--etp",
                "2",
                "--cp",
                "1",
            ]
        )
        self.assertEqual(parsed.node_group_strategy, "ep_pp_dp")
        self.assertEqual(parsed.group_affinity, {0: 128, 1: 128})
        self.assertEqual(parsed.tensor_model_parallel_size, 1)
        self.assertEqual(parsed.pipeline_model_parallel_size, 16)
        self.assertEqual(parsed.expert_model_parallel_size, 32)
        self.assertEqual(parsed.expert_tensor_parallel_size, 2)
        self.assertEqual(parsed.context_parallel_size, 1)

        # the ep_dp_pp rename of the previous contiguous strategy
        parsed = parse_master_args(
            [
                "--job_name",
                "test",
                "--node-group-strategy=ep_dp_pp",
                "--group-affinity={0: 128, 1: 128}",
            ]
        )
        self.assertEqual(parsed.node_group_strategy, "ep_dp_pp")

        # invalid strategy choice -> argparse exits with code 2
        with self.assertRaises(SystemExit) as cm:
            parse_master_args(
                ["--job_name", "test", "--node-group-strategy=bogus"]
            )
        self.assertEqual(cm.exception.code, 2)
