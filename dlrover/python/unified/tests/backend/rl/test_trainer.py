#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List
from unittest import mock
from unittest.mock import patch

import pytest

from dlrover.python.unified.backend.rl.trainer import (
    RoleGroupProxy,
)
from dlrover.python.unified.common.actor_base import ActorInfo, JobInfo
from dlrover.python.unified.common.enums import RLRoleType
from dlrover.python.unified.common.workload_desc import CustomWorkloadDesc
from dlrover.python.unified.tests.backend.rl.classes import (
    SimpleActor,
    SimpleInteractiveTrainer,
)


# Must be async, as ActorBase.__init__ expects an event loop to be running.
@patch("ray.get_actor")
@patch(
    "dlrover.python.unified.backend.rl.trainer.PrimeMasterApi.get_workers_by_role"
)
async def test_basic(mock_get_workers_by_role, mock_get_actor):
    spec = CustomWorkloadDesc(
        module_name=SimpleInteractiveTrainer.__module__,
        class_name=SimpleInteractiveTrainer.__name__,
    )
    actor_spec = CustomWorkloadDesc(
        module_name=SimpleActor.__module__,
        class_name=SimpleActor.__name__,
    )
    mock_roles: Dict[str, List[ActorInfo]] = {
        "TRAINER": [ActorInfo(name="t", role="TRAINER", spec=spec)],
        "ACTOR": [
            ActorInfo(name="a0", role="ACTOR", spec=actor_spec),
            ActorInfo(name="a1", role="ACTOR", spec=actor_spec),
        ],
    }

    def get_workers_by_role(role: str, optional=False) -> List[ActorInfo]:
        if role not in mock_roles and not optional:
            raise ValueError(f"Role {role} not found")
        return mock_roles.get(role, [])

    mock_get_actor.return_value = "actor_handle"
    mock_get_workers_by_role.side_effect = get_workers_by_role

    trainer = SimpleInteractiveTrainer(
        JobInfo(name="test", job_id="test", user_config={"k1": "v1"}),
        ActorInfo(name="test", role="TRAINER", spec=spec),
    )
    trainer._init_role_group_proxy()

    assert isinstance(trainer.RG_ACTOR, RoleGroupProxy)
    assert len(trainer.RG_ACTOR._actor_handles) == 2

    assert len(trainer.actors) == 2
    assert len(trainer.rollouts) == 0
    assert len(trainer.rewards) == 0
    assert len(trainer.references) == 0
    assert len(trainer.critics) == 0
    assert len(trainer.config) == 1


async def test_role_group_proxy(mocker):
    mocker.patch("ray.get", side_effect=lambda x: x)
    mocker.patch("ray.wait", side_effect=lambda *args, **kwargs: (args[0], []))
    mock_get_workers_by_role = mocker.patch(
        "dlrover.python.unified.backend.rl.trainer.PrimeMasterApi.get_workers_by_role"
    )
    mock_get_actor = mocker.patch("ray.get_actor")

    role_group = RoleGroupProxy(RLRoleType.ACTOR.name, 2, SimpleActor, [None])
    assert role_group is not None
    assert role_group.role == RLRoleType.ACTOR.name
    assert role_group.world_size == 2
    assert role_group._can_shard_invocation()
    with pytest.raises(AttributeError):
        role_group.test0()
    with pytest.raises(AttributeError):
        role_group.test1()
    with pytest.raises(AttributeError):
        role_group.test2()
    with pytest.raises(AttributeError):
        role_group.test3()
    with pytest.raises(AttributeError):
        role_group.test4()

    spec = CustomWorkloadDesc(
        module_name=SimpleInteractiveTrainer.__module__,
        class_name=SimpleInteractiveTrainer.__name__,
    )
    actor_spec = CustomWorkloadDesc(
        module_name=SimpleActor.__module__,
        class_name=SimpleActor.__name__,
    )
    mock_roles: Dict[str, List[ActorInfo]] = {
        "TRAINER": [ActorInfo(name="t", role="TRAINER", spec=spec)],
        "ACTOR": [
            ActorInfo(name="a0", role="ACTOR", spec=actor_spec),
            ActorInfo(name="a1", role="ACTOR", spec=actor_spec),
        ],
    }

    def get_workers_by_role(role: str, optional=False) -> List[ActorInfo]:
        if role not in mock_roles and not optional:
            raise ValueError(f"Role {role} not found")
        return mock_roles.get(role, [])

    mock_get_actor.return_value = "actor_handle"
    mock_get_workers_by_role.side_effect = get_workers_by_role

    trainer = SimpleInteractiveTrainer(
        JobInfo(name="test", job_id="test", user_config={}),
        ActorInfo(name="test", role="TRAINER", spec=spec),
    )
    trainer._init_role_group_proxy()

    trainer.RG_ACTOR._actor_handles = mock.MagicMock(
        return_value=[mock.Mock()]
    )
    trainer.RG_ACTOR.test0()
    trainer.RG_ACTOR.test1()
    trainer.RG_ACTOR.test2()
    trainer.RG_ACTOR.test3()
    trainer.RG_ACTOR.test4()

    mocker.patch("dlrover.python.unified.backend.rl.trainer.invoke_actor_t")
    mocker.patch("dlrover.python.unified.backend.rl.trainer.invoke_actors_t")
    trainer.start()
