import time

import pytest
import ray

from dlrover.python.unified.common.workload_config import ElasticWorkloadDesc
from dlrover.python.unified.prime.config import DLConfig, JobConfig
from dlrover.python.unified.prime.master import HybridMaster


@pytest.fixture
def _ray():
    """Fixture to initialize and shutdown Ray."""
    ray.init(
        num_cpus=4,
        num_gpus=0,
        ignore_reinit_error=True,
        namespace="dlrover_hybrid_test",
    )
    yield
    ray.shutdown()


def test_dev_run():
    dl_config = DLConfig(
        workloads={
            "demo": ElasticWorkloadDesc(
                cmd="python -m dlrover.python.hybrid.demo.demo",
                num=2,
                per_node=2,
            )
        },
    )
    config = JobConfig(
        job_name="test_job",
        dl_config=dl_config,
    )
    master = HybridMaster.create(config)
    assert master.status() == "INIT"
    master.start()
    while master.status() != "STOPPED":
        time.sleep(1)


if __name__ == "__main__":
    test_dev_run()
