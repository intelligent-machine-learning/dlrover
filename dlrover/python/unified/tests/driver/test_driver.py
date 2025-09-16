# Copyright 2025 The DLRover Authors. All rights reserved.
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
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from dlrover.python.unified.common.config import JobConfig
from dlrover.python.unified.common.enums import MasterStage
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.driver.main import main, submit
from dlrover.python.unified.tests.fixtures.example_jobs import (
    elastic_training_job,
)


def test_submit():
    fake_master = MagicMock(PrimeMasterApi)
    fake_master.get_status.return_value.stage = MasterStage.STOPPED

    job = elastic_training_job()

    with ExitStack() as stack:
        stack.enter_context(patch("ray.is_initialized", return_value=True))
        mock_create = stack.enter_context(
            patch(
                "dlrover.python.unified.controller.master.PrimeMaster.create"
            )
        )
        mock_create.return_value = fake_master
        ret = submit(job)
        assert mock_create.called
        assert ret is fake_master.get_status.return_value.exit_code


def test_submit_init():
    fake_master = MagicMock(PrimeMasterApi)
    fake_master.get_status.return_value.stage = MasterStage.STOPPED

    job = elastic_training_job()

    with ExitStack() as stack:
        stack.enter_context(patch("ray.is_initialized", return_value=False))
        mock_init = stack.enter_context(patch("ray.init", return_value=None))
        mock_create = stack.enter_context(
            patch(
                "dlrover.python.unified.controller.master.PrimeMaster.create"
            )
        )
        mock_create.return_value = fake_master
        ret = submit(job)
        assert mock_init.called
        assert mock_init.call_args.kwargs["runtime_env"] is not None
        assert (
            mock_init.call_args.kwargs["runtime_env"]["working_dir"]
            is not None
        )

        assert mock_create.called
        assert ret is fake_master.get_status.return_value.exit_code


def test_driver():
    with patch("dlrover.python.unified.driver.main.submit") as mock_submit:
        job = elastic_training_job()

        with pytest.raises(ValueError, match="required"):
            main([])

        main([job.model_dump_json()])
        assert mock_submit.called
        config = mock_submit.call_args.kwargs["config"]

        assert isinstance(config, JobConfig)
        assert config == job
