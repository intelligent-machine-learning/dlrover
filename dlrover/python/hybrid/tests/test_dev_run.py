import time

import ray

from dlrover.python.hybrid.center.config import (
    DLConfig,
    JobConfig,
    WorkloadDesc,
)
from dlrover.python.hybrid.center.master import HybridMaster


def test_dev_run():
    dl_config = DLConfig(
        workloads={
            "demo": WorkloadDesc(
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
    while master.status() != "RUNNING":
        time.sleep(1)
    time.sleep(5)  # wait for the job to start
    master.stop()
    while master.status() != "STOPPED":
        time.sleep(1)

    ray.shutdown()


if __name__ == "__main__":
    test_dev_run()
