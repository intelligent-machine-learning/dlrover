# Copyright 2026 The DLRover Authors. All rights reserved.
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
from dlrover.brain.python.jobmanagement.job_config import (
    JobConfig,
    JobConfigScope,
    JobConfigValues,
)
from dlrover.brain.python.jobmanagement.job_config_manager import (
    JobConfigManager,
)
from dlrover.brain.python.common.job import JobMeta


class TestJobConfigSystem(unittest.TestCase):
    def test_job_config_scope(self):
        """Test if the scope correctly matches JobMeta attributes."""
        job = JobMeta(
            user="alice", namespace="default", cluster="prod", app="training"
        )

        # Case 1: Empty scope should match everything
        empty_scope = JobConfigScope()
        self.assertTrue(empty_scope.in_scope(job))

        # Case 2: Exact match on one field
        user_scope = JobConfigScope({"user": ["alice", "bob"]})
        self.assertTrue(user_scope.in_scope(job))

        # Case 3: Mismatch on one field
        wrong_user_scope = JobConfigScope({"user": ["bob", "charlie"]})
        self.assertFalse(wrong_user_scope.in_scope(job))

        # Case 4: Multiple conditions (Acts as an AND gate)
        multi_scope = JobConfigScope({"user": ["alice"], "cluster": ["prod"]})
        self.assertTrue(multi_scope.in_scope(job))

        multi_scope_wrong = JobConfigScope(
            {"user": ["alice"], "cluster": ["dev"]}
        )
        self.assertFalse(multi_scope_wrong.in_scope(job))

    def test_job_config_include_exclude_logic(self):
        """Test how JobConfig handles include and exclude combinations."""
        job = JobMeta(user="alice", cluster="prod")

        inc_scope = JobConfigScope({"user": ["alice"]})
        exc_scope = JobConfigScope({"cluster": ["prod"]})

        # Case 1: Include only
        config_inc = JobConfig(include_scope=inc_scope)
        self.assertTrue(config_inc.in_scope(job))

        # Case 2: Exclude only (Matches if it doesn't trigger the exclude scope)
        config_exc = JobConfig(exclude_scope=exc_scope)
        self.assertFalse(
            config_exc.in_scope(job)
        )  # Excluded because cluster="prod"

        # Case 3: Both Include and Exclude (Exclude overrides Include)
        config_both = JobConfig(
            include_scope=inc_scope, exclude_scope=exc_scope
        )
        self.assertFalse(config_both.in_scope(job))

    def test_job_config_manager_resolution(self):
        """Test if the manager finds the first matching config."""
        manager = JobConfigManager()

        # Setup Config 1 (For 'bob')
        c1 = JobConfig(include_scope=JobConfigScope({"user": ["bob"]}))
        c1._config_values = JobConfigValues(
            {"memory": "4G"}
        )  # Manually setting for test

        # Setup Config 2 (For 'alice')
        c2 = JobConfig(include_scope=JobConfigScope({"user": ["alice"]}))
        c2._config_values = JobConfigValues({"memory": "8G"})

        # Setup Config 3 (Fallback/Default, no scope)
        c3 = JobConfig()
        c3._config_values = JobConfigValues({"memory": "1G"})

        # Load into manager
        manager._configs = [c1, c2, c3]

        # Case 1: Match config 2
        job_alice = JobMeta(user="alice")
        res_alice = manager.get_job_config(job_alice)
        self.assertIsNotNone(res_alice)
        self.assertEqual(res_alice._configs["memory"], "8G")

        # Case 2: Match config 1
        job_bob = JobMeta(user="bob")
        res_bob = manager.get_job_config(job_bob)
        self.assertIsNotNone(res_bob)
        self.assertEqual(res_bob._configs["memory"], "4G")

        # Case 3: Fallthrough to config 3 (Default)
        job_charlie = JobMeta(user="charlie")
        res_charlie = manager.get_job_config(job_charlie)
        self.assertIsNotNone(res_charlie)
        self.assertEqual(res_charlie._configs["memory"], "1G")

    def test_job_config_manager_no_match(self):
        """Test when absolutely no configs match."""
        manager = JobConfigManager()

        c1 = JobConfig(include_scope=JobConfigScope({"user": ["admin"]}))
        manager._configs = [c1]

        job_guest = JobMeta(user="guest")
        res_guest = manager.get_job_config(job_guest)

        # Should return None because 'guest' doesn't match 'admin' and there is no fallback
        self.assertIsNone(res_guest)


if __name__ == "__main__":
    unittest.main()
