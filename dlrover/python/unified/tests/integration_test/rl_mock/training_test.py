import pytest

from dlrover.python.unified.api.builder.base import DLJob
from dlrover.python.unified.api.builder.rl import RLJobBuilder
from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.common.workload_desc import (
    ElasticWorkloadDesc,
    ResourceDesc,
    SimpleWorkloadDesc,
)
from dlrover.python.unified.controller.config import (
    ACCELERATOR_TYPE,
    DLConfig,
    JobConfig,
)
from dlrover.python.unified.controller.master import PrimeMaster

classes = f"{__package__}.classes"


def test_run(tmp_ray):
    workloads = {}
    workloads[RLRoleType.ACTOR.name] = ElasticWorkloadDesc(
        total=2,
        per_group=2,
        entry_point=f"{classes}.Actor",
        resource=ResourceDesc(accelerator=0.5),
    )
    workloads[RLRoleType.CRITIC.name] = ElasticWorkloadDesc(
        total=1,
        entry_point=f"{classes}.Critic",
        resource=ResourceDesc(accelerator=1),
    )
    workloads[RLRoleType.ROLLOUT.name] = SimpleWorkloadDesc(
        total=2,
        entry_point=f"{classes}.Rollout",
        resource=ResourceDesc(accelerator=0.5),
    )
    workloads[RLRoleType.REFERENCE.name] = ElasticWorkloadDesc(
        total=1,
        entry_point=f"{classes}.Reference",
        resource=ResourceDesc(accelerator=1),
    )
    workloads[RLRoleType.REWARD.name] = ElasticWorkloadDesc(
        total=1,
        entry_point=f"{classes}.Reward",
        resource=ResourceDesc(accelerator=1),
    )
    workloads["trainer"] = SimpleWorkloadDesc(
        resource=ResourceDesc(cpu=1),
        entry_point=f"{classes}.Trainer",
    )
    config = DLConfig(
        user_config={},
        workloads=workloads,
        accelerator_type=ACCELERATOR_TYPE.CPU,
    )

    master = PrimeMaster.create(
        JobConfig(job_name="openrlhf_demo", dl_config=config)
    )
    master.start()
    master.wait()
    assert master.get_status().exit_code == 0, master.get_status()
    master.shutdown()


@pytest.mark.skip(reason="API not implemented yet")
def test_api(tmp_ray):
    rl_job: DLJob = (
        RLJobBuilder()
        .node_num(2)
        .device_per_node(4)
        .device_type("CPU")
        .config({})
        .trainer(classes, "Trainer")
        .resource(cpu=1)
        .actor(classes, "Actor")
        .total(2)
        .per_group(1)
        .rollout(classes, "Rollout")
        .total(2)
        .per_group(1)
        .reference(classes, "Reference")
        .total(2)
        .per_group(1)
        .build()
    )

    ret = rl_job.submit("openrlhf_demo", master_cpu=1, master_memory=128)
    assert ret == 0
