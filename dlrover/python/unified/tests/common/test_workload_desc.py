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

import pytest
from pydantic import ValidationError

from dlrover.python.unified.common.enums import WorkloadEntrypointType
from dlrover.python.unified.common.workload_desc import (
    ElasticWorkloadDesc,
    ResourceDesc,
    SimpleWorkloadDesc,
    get_entrypoint_type,
)


def test_normal():
    assert SimpleWorkloadDesc(entry_point="a.b.c").backend == "simple"
    assert ElasticWorkloadDesc(entry_point="a.b.c").backend == "elastic"


def test_validate():
    assert SimpleWorkloadDesc(entry_point="a::b").entry_point == "a.b"
    with pytest.raises(ValidationError, match="entry_point"):
        SimpleWorkloadDesc(entry_point="a_b_c")

    with pytest.raises(ValidationError, match="Resource must not be empty"):
        SimpleWorkloadDesc(entry_point="a.b", resource=ResourceDesc())

    with pytest.raises(ValidationError, match="divisible"):
        SimpleWorkloadDesc(total=5, per_group=3, entry_point="a.b")

    # Current usage
    SimpleWorkloadDesc(
        total=4,
        per_group=2,
        entry_point="a.b",
        rank_based_gpu_selection=True,
    )
    # per_group == 1
    with pytest.raises(ValidationError, match="rank_based_gpu_selection"):
        SimpleWorkloadDesc(entry_point="a.b", rank_based_gpu_selection=True)
    # gpu > 1
    with pytest.raises(ValidationError, match="rank_based_gpu_selection"):
        SimpleWorkloadDesc(
            total=4,
            per_group=2,
            entry_point="a.b",
            rank_based_gpu_selection=True,
            resource=ResourceDesc(accelerator=2),
        )


def test_get_entrypoint_type():
    assert (
        get_entrypoint_type("test.test") == WorkloadEntrypointType.MODULE_FUNC
    )
    assert (
        get_entrypoint_type("my.module.run")
        == WorkloadEntrypointType.MODULE_FUNC
    )
    assert get_entrypoint_type("test.py") == WorkloadEntrypointType.PY_CMD
    assert (
        get_entrypoint_type("test.py --test1") == WorkloadEntrypointType.PY_CMD
    )
    assert get_entrypoint_type("./test.py") == WorkloadEntrypointType.PY_CMD
    assert (
        get_entrypoint_type("/path/test.py") == WorkloadEntrypointType.PY_CMD
    )
    assert get_entrypoint_type("test") is None
