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

import asyncio
from concurrent.futures import Future
from threading import Thread
from typing import Sequence
from unittest.mock import AsyncMock, Mock, patch

import pytest
import ray
from torch.utils.data import Dataset

from dlrover.python.unified.api.runtime.queue import DataQueue
from dlrover.python.unified.api.runtime.ray_dataloader_iter import (
    patch_dataloader_ray,
)
from dlrover.python.unified.api.runtime.rpc_helper import (
    RPC_REGISTRY,
    RoleGroup,
    create_rpc_proxy,
    export_rpc_instance,
    export_rpc_method,
    rpc,
)
from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.common.actor_base import ActorInfo
from dlrover.python.unified.util import async_helper
from dlrover.python.unified.util.actor_helper import ActorInvocation


def mock_rpc_call(actor_name, method, args, kwargs):
    actor = Mock(BaseWorker)
    actor.actor_info = Mock(ActorInfo)
    actor.actor_info.name = actor_name
    actor._arbitrary_remote_call = BaseWorker._arbitrary_remote_call.__get__(
        actor
    )
    actor._user_rpc_ready = asyncio.Event()
    actor._user_rpc_ready.set()
    ret = BaseWorker._user_rpc_call(actor, method, *args, **kwargs)
    v = asyncio.run(ret)
    ret = Mock(ActorInvocation)
    ret.pending = False
    ret.wait.return_value = v
    ret.async_wait = AsyncMock(return_value=v)
    return ret


def mock_as_future(co):
    future = Future()
    Thread(
        target=lambda: future.set_result(asyncio.run(co)), daemon=True
    ).start()
    return future


def test_rpc(mocker):
    mocker.patch(
        "dlrover.python.unified.api.runtime.rpc_helper._rpc_call",
        mock_rpc_call,
    )
    RPC_REGISTRY.clear()

    # Need export explicitly, as it's not top-level.
    @rpc(export=True)
    def some_method():
        return "test1"

    @rpc(export=True)
    async def some_async_method():
        return "test2"

    # stub
    def foo_method(x, y=0): ...

    @rpc(foo_method, export=True)
    def foo_method_impl(x, y=0):
        return f"{x}-{y}"

    assert RPC_REGISTRY["some_method"] == some_method
    assert RPC_REGISTRY["some_async_method"] == some_async_method
    assert RPC_REGISTRY["foo_method"] == foo_method_impl
    with pytest.raises(
        ValueError, match="RPC method 'some_method' already registered."
    ):
        export_rpc_method("some_method", some_method)
    RPC_REGISTRY.clear()


def test_rpc_proxy(mocker):
    mocker.patch(
        "dlrover.python.unified.api.runtime.rpc_helper._rpc_call",
        mock_rpc_call,
    )

    class SimpleClass:
        @rpc()
        def hello(self, name: str) -> str:
            return f"Hello, {name}!"

    export_rpc_instance("simple_class", SimpleClass())
    assert "simple_class.hello" in RPC_REGISTRY

    proxy = create_rpc_proxy("actor", "simple_class", SimpleClass)
    assert proxy.hello("World") == "Hello, World!"


def test_queue(mocker):
    register_data_queue = mocker.patch(
        "dlrover.python.unified.controller.api.PrimeMasterApi.register_data_queue",
        return_value=None,
    )
    mocker.patch(
        "dlrover.python.unified.controller.api.PrimeMasterApi.get_data_queue_owner",
        return_value="actor",
    )
    mocker.patch(
        "dlrover.python.unified.api.runtime.rpc_helper._rpc_call",
        mock_rpc_call,
    )
    mocker.patch(
        "dlrover.python.unified.api.runtime.queue.as_future",
        mock_as_future,
    )
    mocker.patch(
        "dlrover.python.unified.api.runtime.queue.wait",
        asyncio.run,
    )
    BaseWorker.CURRENT = Mock()
    # Mock ray.put, ray.get, not to actually use Ray.
    mocker.patch("ray.put", side_effect=lambda x: x)
    mocker.patch("ray.get", side_effect=lambda x: x)

    BaseWorker.CURRENT.actor_info.name = "actor1"
    queue_master = DataQueue[str]("test_queue", size=10, is_master=True)
    assert register_data_queue.called_once_with("test_queue", "actor1", 10)
    assert "DataQueue.test_queue.qsize" in RPC_REGISTRY

    BaseWorker.CURRENT.actor_info.rank = 1
    queue_client = DataQueue[str]("test_queue", size=10, is_master=False)

    assert queue_client.qsize() == 0
    queue_master.put("item1")
    assert queue_client.qsize() == 1
    assert queue_client.get(1) == ["item1"]
    queue_client.put("item2")
    assert queue_master.get_nowait(1) == ["item2"]


def make_actor_info(name, rank):
    info: ActorInfo = Mock(ActorInfo)
    info.name = name
    info.rank = rank
    return info


@pytest.fixture
def setup_async_helper():
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    async_helper.__main_loop = loop
    yield
    async_helper.__main_loop = None
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()


def test_role_group_basic(mocker, setup_async_helper):
    def fake_rpc_call(actor, method, args, kwargs):
        ret = Mock()
        ret.pending = False
        ret.result_or_exception = f"{actor}-{method}-{args}-{kwargs}"
        ret.async_wait = AsyncMock(return_value=ret.result_or_exception)
        return ret

    # Mock PrimeMasterApi.get_workers_by_role
    mocker.patch(
        "dlrover.python.unified.controller.api.PrimeMasterApi.get_workers_by_role",
        return_value=[
            make_actor_info("actorA", 0),
            make_actor_info("actorB", 1),
        ],
    )
    # Mock _rpc_call to return a mock with async_wait/wait
    mocker.patch(
        "dlrover.python.unified.api.runtime.rpc_helper._rpc_call",
        fake_rpc_call,
    )

    group = RoleGroup("worker")
    assert len(group) == 2
    assert group[0].name == "actorA"

    # Test call with method name
    fut = group.call("foo_method")
    assert fut.result() == [
        "actorA-foo_method-()-{}",
        "actorB-foo_method-()-{}",
    ]

    # stub
    def foo_method(x, y=0): ...

    fut = group.call(foo_method, 5, y=10)
    assert fut.result() == [
        "actorA-foo_method-(5,)-{'y': 10}",
        "actorB-foo_method-(5,)-{'y': 10}",
    ]

    # Test call with scatter
    fut = group.call(foo_method, [5, 10], y=[10, 20], _scatter=True)
    assert fut.result() == [
        "actorA-foo_method-(5,)-{'y': 10}",
        "actorB-foo_method-(10,)-{'y': 20}",
    ]

    # Test call_rank0
    fut3 = group.call_rank0(foo_method, 3, y=4)
    assert fut3.result() == "actorA-foo_method-(3,)-{'y': 4}"


def test_role_group_optional(mocker):
    mocker.patch(
        "dlrover.python.unified.controller.api.PrimeMasterApi.get_workers_by_role",
        side_effect=ValueError("not found"),
    )
    group = RoleGroup("not_exist", optional=True)
    assert len(group) == 0


def test_role_group_no_actors(mocker):
    mocker.patch(
        "dlrover.python.unified.controller.api.PrimeMasterApi.get_workers_by_role",
        return_value=[],
    )

    def f(x) -> Sequence: ...  # type: ignore

    with pytest.raises(ValueError, match="No actors found for role 'empty'"):
        RoleGroup("empty")
    with pytest.raises(ValueError, match="No actors in the role group."):
        RoleGroup("empty", optional=True).call_batch(f, 4, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="No actors in the role group."):
        RoleGroup("empty", optional=True).call_rank0(f, [])

    assert (
        RoleGroup("empty", optional=True).call(f, 4).result() == []
    )  # allow call


def test_rolegroup_call_batch(mocker, setup_async_helper):
    mocker.patch(
        "dlrover.python.unified.controller.api.PrimeMasterApi.get_workers_by_role",
        return_value=[
            make_actor_info("actorA", 0),
            make_actor_info("actorB", 1),
        ],
    )

    def fake_rpc_call(actor, method, args, kwargs):
        v = [f"{actor}-{it}" for it in args[0]]

        ret = Mock()
        ret.pending = False
        ret.result_or_exception = v
        ret.async_wait = AsyncMock(return_value=v)
        return ret

    mocker.patch(
        "dlrover.python.unified.api.runtime.rpc_helper._rpc_call",
        fake_rpc_call,
    )
    group = RoleGroup("worker")
    # 2 actors, batch size 4, args is a sequence
    fut_seq = group.call_batch(lambda x: x, 4, [1, 2, 3, 4])
    # Should split into 2 batches of 2
    assert len(fut_seq) == 4
    # Each result should be from the corresponding actor
    results = fut_seq.wait()
    assert results == ["actorA-1", "actorA-2", "actorB-3", "actorB-4"]
    assert list(fut_seq) == results
    assert list(fut_seq[1:2]) == results[1:2]
    assert repr(fut_seq) == "FutureSequence(lens=[2, 2])"


def test_ray_dataloader(shared_ray):
    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            assert ray.get_runtime_context().current_actor is not None, (
                "Dataset must be used in a Ray actor."
            )
            return self.data[idx]

    # Auto recover
    with patch("torch.utils.data.DataLoader._get_iterator"):
        patch_dataloader_ray()
        dataset = SimpleDataset(list(range(100)))
        from torch.utils.data import DataLoader

        ret = list(DataLoader(dataset, batch_size=10))
        assert len(ret) == 10 and all(len(batch) == 10 for batch in ret)
