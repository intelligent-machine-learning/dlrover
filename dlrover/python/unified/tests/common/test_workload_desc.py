import pytest
from pydantic import ValidationError

from dlrover.python.unified.common.workload_desc import (
    ElasticWorkloadDesc,
    ResourceDesc,
    SimpleWorkloadDesc,
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
