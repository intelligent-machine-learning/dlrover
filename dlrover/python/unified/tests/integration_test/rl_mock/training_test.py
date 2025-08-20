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


def test_run(tmp_ray):
    workloads = {}
    workloads[RLRoleType.ACTOR.name] = ElasticWorkloadDesc(
        total=2,
        per_group=2,
        entry_point=f"{__package__}.classes.Actor",
        resource=ResourceDesc(accelerator=0.5),
    )
    workloads[RLRoleType.CRITIC.name] = ElasticWorkloadDesc(
        total=1,
        entry_point=f"{__package__}.classes.Critic",
        resource=ResourceDesc(accelerator=1),
    )
    workloads[RLRoleType.ROLLOUT.name] = SimpleWorkloadDesc(
        total=2,
        entry_point=f"{__package__}.classes.Rollout",
        resource=ResourceDesc(accelerator=0.5),
    )
    workloads[RLRoleType.REFERENCE.name] = ElasticWorkloadDesc(
        total=1,
        entry_point=f"{__package__}.classes.Reference",
        resource=ResourceDesc(accelerator=1),
    )
    workloads[RLRoleType.REWARD.name] = ElasticWorkloadDesc(
        total=1,
        entry_point=f"{__package__}.classes.Reward",
        resource=ResourceDesc(accelerator=1),
    )
    workloads["trainer"] = SimpleWorkloadDesc(
        resource=ResourceDesc(cpu=1),
        entry_point=f"{__package__}.classes.Trainer",
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
