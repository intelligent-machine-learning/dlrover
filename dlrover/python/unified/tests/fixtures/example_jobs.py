from dlrover.python.unified.common.workload_config import ElasticWorkloadDesc
from dlrover.python.unified.controller.config import (
    ACCELERATOR_TYPE,
    DLConfig,
    JobConfig,
)


def elastic_training_job():
    """Example job configuration for elastic training."""

    dl_config = DLConfig(
        workloads={
            "training": ElasticWorkloadDesc(
                cmd="python -m dlrover.trainer.torch.node_check.nvidia_gpu",
                instance_number=2,
                proc_per_worker=2,
            )
        },
        accelerator_type=ACCELERATOR_TYPE.CPU,
    )
    return JobConfig(
        job_name="test_elastic_training",
        dl_config=dl_config,
    )
