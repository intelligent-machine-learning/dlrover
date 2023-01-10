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

import os
import unittest

from dlrover.trainer.tensorflow.executor.base_executor import BaseExecutor


class BaseExecutorTest(unittest.TestCase):
    def test_get_cluster_info_by_tf_config(self):
        os.environ[
            "TF_CONFIG"
        ] = '{"cluster": { \
            "ps": ["web04-pod2.default.svc:5002"],\
            "chief":["p1.default.svc:5001"],\
            "worker": ["web04-pod1.default.svc:5000"]},\
            "task": {"type": "worker", "index": 0}}'
        base_executor = BaseExecutor()
        base_executor.get_cluster_info_by_tf_config()
        cluster_spec = {
            "ps": ["web04-pod2.default.svc:5002"],
            "chief": ["p1.default.svc:5001"],
            "worker": ["web04-pod1.default.svc:5000"],
        }
        self.assertDictEqual(cluster_spec, base_executor.cluster_spec)

    def test_get_cluster_def(self):
        cluster_spec = {
            "ps": ["web04-pod2.default.svc:5002"],
            "chief": ["p1.default.svc:5001"],
            "worker": ["web04-pod1.default.svc:5000"],
        }
        base_executor = BaseExecutor()
        base_executor.task_type = "worker"
        base_executor.task_id = 0
        base_executor.role = "worker:0"
        base_executor.address = "web04-pod1.default.svc:5000"
        cluster_def = base_executor.get_cluster_def(cluster_spec)
        mini_cluster_spec = {
            "ps": ["web04-pod2.default.svc:5002"],
            "worker": ["web04-pod1.default.svc:5000"],
        }
        self.assertDictEqual(
            mini_cluster_spec, base_executor.mini_cluster_spec
        )
        self.assertEqual(cluster_def.job[0].name, "ps")
        self.assertEqual(len(cluster_def.job[0].tasks), 1)
        self.assertEqual(cluster_def.job[1].name, "worker")
        self.assertEqual(len(cluster_def.job[1].tasks), 1)
        self.assertEqual(
            cluster_def.job[1].tasks[1], "web04-pod1.default.svc:5000"
        )

    def test_get_chief_config(self):
        os.environ[
            "TF_CONFIG"
        ] = '{"cluster": { \
            "ps": ["web04-pod2.default.svc:5002"],\
            "chief":["p1.default.svc:5001"],\
            "worker": ["web04-pod1.default.svc:5000"]},\
            "task": {"type": "chief", "index": 0}}'
        base_executor = BaseExecutor()
        base_executor.get_cluster_info_by_tf_config()
        cluster_spec = {
            "ps": ["web04-pod2.default.svc:5002"],
            "chief": ["p1.default.svc:5001"],
            "worker": ["web04-pod1.default.svc:5000"],
        }
        estimator_run_config = base_executor.get_config(cluster_spec)
        sess_config = estimator_run_config._session_config
        self.assertEqual(
            estimator_run_config._master, "grpc://p1.default.svc:5001"
        )

        self.assertEqual(
            sess_config.experimental.share_session_state_in_clusterspec_propagation,  # noqa: E501
            True,
        )
        self.assertEqual(estimator_run_config._task_type, "chief")
        self.assertEqual(estimator_run_config._task_id, 0)

    def test_get_worker_config(self):
        os.environ[
            "TF_CONFIG"
        ] = '{"cluster": { \
            "ps": ["web04-pod2.default.svc:5002"],\
            "chief":["p1.default.svc:5001"],\
            "worker": ["web04-pod1.default.svc:5000"]},\
            "task": {"type": "worker", "index": 0}}'
        base_executor = BaseExecutor()
        base_executor.get_cluster_info_by_tf_config()
        cluster_spec = {
            "ps": ["web04-pod2.default.svc:5002"],
            "chief": ["p1.default.svc:5001"],
            "worker": ["web04-pod1.default.svc:5000"],
        }
        estimator_run_config = base_executor.get_config(cluster_spec)
        sess_config = estimator_run_config._session_config
        self.assertEqual(
            estimator_run_config._master, "grpc://web04-pod1.default.svc:5000"
        )
        self.assertEqual(
            sess_config.experimental.share_session_state_in_clusterspec_propagation,  # noqa: E501
            True,
        )
        self.assertEqual(estimator_run_config._task_type, "worker")
        self.assertEqual(estimator_run_config._task_id, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
