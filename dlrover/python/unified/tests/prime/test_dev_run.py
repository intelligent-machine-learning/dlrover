import time

import pytest
import ray

from dlrover.python.unified.common.workload_config import ElasticWorkloadDesc
from dlrover.python.unified.prime.config import DLConfig, JobConfig
from dlrover.python.unified.prime.master import PrimeMaster


@pytest.fixture
def _ray():
    """Fixture to initialize and shutdown Ray."""
    ray.init(
        num_cpus=4,
        num_gpus=0,
        ignore_reinit_error=True,
        namespace="dlrover_unified_test",
        runtime_env={
            "env_vars": {
                "COVERAGE_PROCESS_START": ".coveragerc",
            }
        },
    )
    yield
    time.sleep(2)
    ray.shutdown()

    import coverage

    coverage.Coverage().combine()


def test_dev_run(_ray):
    dl_config = DLConfig(
        workloads={
            "demo": ElasticWorkloadDesc(
                cmd="python -m dlrover.trainer.torch.node_check.nvidia_gpu",
                num=2,
                per_node=2,
            )
        },
    )
    config = JobConfig(
        job_name="test_job",
        dl_config=dl_config,
    )
    master = PrimeMaster.create(config)
    assert master.status() == "INIT"
    master.start()
    while master.status() != "STOPPED":
        time.sleep(1)
    master.shutdown()
