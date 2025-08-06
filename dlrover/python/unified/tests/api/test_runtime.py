import asyncio
from concurrent.futures import Future
from threading import Thread
from unittest.mock import Mock

from dlrover.python.unified.api.runtime.queue import DataQueue
from dlrover.python.unified.api.runtime.rpc import (
    RPC_REGISTRY,
    create_rpc_proxy,
    export_rpc_instance,
    rpc,
)
from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.common.workload_base import ActorInfo, WorkerStage


def mock_rpc_call(actor_name, method, is_async, args, kwargs):
    actor = Mock(BaseWorker)
    actor.actor_info = Mock(ActorInfo)
    actor.actor_info.name = actor_name
    actor._arbitrary_remote_call = BaseWorker._arbitrary_remote_call.__get__(
        actor
    )
    actor.stage = WorkerStage.RUNNING
    ret = BaseWorker._user_rpc_call(actor, method, *args, **kwargs)
    if not is_async:
        return asyncio.run(ret)
    return ret


def mock_as_future(co):
    future = Future()
    Thread(
        target=lambda: future.set_result(asyncio.run(co)), daemon=True
    ).start()
    return future


def test_rpc(mocker):
    mocker.patch(
        "dlrover.python.unified.api.runtime.rpc._rpc_call",
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

    assert RPC_REGISTRY["some_method"] == some_method
    assert RPC_REGISTRY["some_async_method"] == some_async_method


def test_rpc_proxy(mocker):
    mocker.patch(
        "dlrover.python.unified.api.runtime.rpc._rpc_call",
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
        "dlrover.python.unified.api.runtime.rpc._rpc_call",
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
