#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from unittest.mock import MagicMock

from dlrover.python.common.constants import TrainingExceptionLevel
from dlrover.python.unified.common.args import parse_job_args
from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.constant import InternalDLWorkloadRole
from dlrover.python.unified.common.dl_context import DLContext
from dlrover.python.unified.common.failure import FailureDesc
from dlrover.python.unified.common.job_context import get_job_context
from dlrover.python.unified.master.elastic.failover import (
    FAILURE_TYPE_KEY,
    ElasticFailoverCoordinator,
)
from dlrover.python.unified.master.elastic.job_manager import ElasticJobManager
from dlrover.python.unified.tests.base import RayBaseTest
from dlrover.python.unified.tests.test_data import TestData


class ElasticFailoverCoordinatorTest(RayBaseTest):
    def setUp(self):
        super().setUp()
        args = [
            "--job_name",
            "test",
            "--dl_type",
            "SFT",
            "--dl_config",
            f"{TestData.UD_SIMPLE_TEST_SFT_CONF_0}",
        ]
        parsed_args = parse_job_args(args)
        job_config = JobConfig.build_from_args(parsed_args)
        dl_context = DLContext.build_from_args(parsed_args)

        self._job_context = get_job_context()
        self._job_context.init(job_config, dl_context)

    def tearDown(self):
        self.close_ray_safely()
        super().tearDown()

    def test_handle_failures(self):
        job_manager = ElasticJobManager()
        job_manager.re_execute = MagicMock(return_value=None)

        def callback():
            return

        fc = ElasticFailoverCoordinator(job_manager, callback, callback)

        desc = FailureDesc(
            workload_name="ELASTIC_1-0_1-0",
            workload_role=InternalDLWorkloadRole.ELASTIC_ROLE,
            failure_time=int(time.time()),
            failure_level=3,
            reason="unknown",
            extra_info={FAILURE_TYPE_KEY: TrainingExceptionLevel.NODE_ERROR},
        )

        fc.handle_failures([desc])
        job_manager.re_execute.assert_called_once()

        desc = FailureDesc(
            workload_name="ELASTIC_1-0_1-0",
            workload_role=InternalDLWorkloadRole.ELASTIC_ROLE,
            failure_time=int(time.time()),
            failure_level=3,
            reason="unknown",
        )

        fc.handle_failures([desc])
        job_manager.re_execute.assert_called_once()
