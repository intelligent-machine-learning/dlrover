import time
import ray
from dlrover.python.hybrid.config import DLConfig, JobConfig, TrainerDesc, WorkloadDesc
from dlrover.python.hybrid.master import HybridMaster


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
    master = HybridMaster.remote(config)
    assert ray.get(master.status.remote()) == "INIT"
    ray.get(master.start.remote())
    while ray.get(master.status.remote()) != "RUNNING":
        time.sleep(1)
    time.sleep(5)  # wait for the job to start
    ray.get(master.stop.remote())
    while ray.get(master.status.remote()) != "STOPPED":
        time.sleep(1)

    ray.shutdown()


if __name__ == "__main__":
    test_dev_run()
