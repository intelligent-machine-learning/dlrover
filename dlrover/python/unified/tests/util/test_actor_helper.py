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
import random
from typing import Any
from unittest.mock import patch

import pytest
import ray
from pytest_mock import MockerFixture
from ray.actor import ActorClass

import dlrover.python.unified.util.actor_helper as ah
from dlrover.python.unified.util.actor_helper import (
    actor_call,
    invoke_actor,
    invoke_actors,
    invoke_meta,
    wait_batch_invoke,
)


@ray.remote
class SimpleActor:
    def some_method(self):
        return "ok"

    def some_method_with_arg(self, a: int, b: str):
        assert a == 1 and b == "b"
        return "ok"

    def method_exception(self):
        raise Exception("This is a test exception")

    def slow_method(self, time: float):
        """A method that simulates a slow operation."""
        import time as time_module

        time_module.sleep(time)
        return "done"

    def method_relaunch(self):
        if not ray.get_runtime_context().was_current_actor_reconstructed:
            ray.kill(ray.get_runtime_context().current_actor, no_restart=False)
        return "restarted"

    def is_restarted(self):
        return ray.get_runtime_context().was_current_actor_reconstructed


class Stub:
    def __init__(self, actor: str):
        self.ACTOR_NAME = actor

    @actor_call
    @staticmethod
    def some_method() -> str: ...  # type: ignore[empty-body]
    @actor_call
    @staticmethod
    def some_method_with_arg(a: int, b: str) -> str: ...  # type: ignore[empty-body]
    @actor_call
    @staticmethod
    def slow_method(time: float) -> str: ...  # type: ignore[empty-body]
    @actor_call
    @staticmethod
    def method_relaunch() -> str: ...  # type: ignore[empty-body]
    @actor_call
    @staticmethod
    def method_exception() -> str: ...  # type: ignore[empty-body]
    @actor_call
    @staticmethod
    def is_restarted() -> bool: ...  # type: ignore[empty-body]

    @actor_call
    @staticmethod
    def not_existent_method() -> str: ...  # type: ignore[empty-body]

    @actor_call
    @staticmethod
    @invoke_meta(name="some_method")
    def some_method_alias() -> str: ...  # type: ignore[empty-body]

    @actor_call
    @staticmethod
    @invoke_meta(name="__ray_terminate__")
    def terminate() -> None: ...  # type: ignore[empty-body]


@pytest.fixture
def tmp_actor1(shared_ray):
    name = f"actor1_{random.randint(1, 10000)}"
    actor = SimpleActor.options(name=name, max_restarts=-1).remote()
    ah.__actors_cache.clear()
    yield name
    ah.kill_actors([name])
    del actor


@pytest.fixture
def tmp_actor2(shared_ray):
    name = f"actor2_{random.randint(1, 10000)}"
    actor = SimpleActor.options(name=name, max_restarts=-1).remote()
    yield name
    ah.kill_actors([name])
    del actor


def test_get_actor_with_cache(tmp_actor1):
    ah.__actors_cache.clear()
    actor = ah.get_actor_with_cache(tmp_actor1)
    assert actor is not None
    with pytest.raises(ValueError):
        ah.get_actor_with_cache("actor_not_found")

    with patch("ray.get_actor") as mock_get_actor:
        actor2 = ah.get_actor_with_cache(tmp_actor1)
        assert actor2 is actor
        mock_get_actor.assert_not_called()


def test_as_actor_class():
    class Actor:
        def some_method(self):
            return "ok"

    actor_class = ah.as_actor_class(Actor)
    assert isinstance(actor_class, ActorClass)
    actor_class: Any = ray.remote(Actor)
    assert ah.as_actor_class(actor_class) is actor_class
    assert isinstance(actor_class, ActorClass)


def test_invoke_actor(tmp_actor1):
    with pytest.raises(ValueError):
        invoke_actor(Stub.some_method, "non_existent_actor").wait()
    assert invoke_actor(Stub.some_method, tmp_actor1).wait() == "ok"
    assert invoke_actor(Stub.some_method_alias, tmp_actor1).wait() == "ok"
    assert (
        invoke_actor(Stub.some_method_with_arg, tmp_actor1, 1, b="b").wait()
        == "ok"
    )

    with pytest.raises(AttributeError):
        invoke_actor(Stub.not_existent_method, tmp_actor1).wait()
    assert invoke_actor(Stub.method_relaunch, tmp_actor1).wait() == "restarted"
    with pytest.raises(Exception, match="This is a test exception"):
        invoke_actor(Stub.method_exception, tmp_actor1).wait()

    # No ActorDieError when shutdown
    assert invoke_actor(Stub.terminate, tmp_actor1).wait() is None


def test_invoke_actor_async(tmp_actor1):
    def async_wait(invoke: ah.ActorInvocation):
        return asyncio.run(invoke.async_wait())

    with pytest.raises(ValueError):
        async_wait(invoke_actor(Stub.some_method, "non_existent_actor"))
    assert async_wait(invoke_actor(Stub.some_method, tmp_actor1)) == "ok"
    with pytest.raises(AttributeError):
        async_wait(invoke_actor(Stub.not_existent_method, tmp_actor1))

    assert (
        async_wait(invoke_actor(Stub.method_relaunch, tmp_actor1))
        == "restarted"
    )
    with pytest.raises(Exception, match="This is a test exception"):
        async_wait(invoke_actor(Stub.method_exception, tmp_actor1))


async def test_invoke_actors(tmp_actor1, tmp_actor2):
    result = await invoke_actors(Stub.some_method, [tmp_actor1, tmp_actor2])
    assert result.is_all_successful
    assert result[0] == "ok"
    assert result[1] == "ok"


async def test_invoke_actors_hang(
    mocker: MockerFixture, tmp_actor1, tmp_actor2
):
    """Mock scene, one worker exception, cause another hang"""
    mocker.patch.object(
        ah.ActorBatchInvocation.async_wait,
        "__defaults__",
        (0.1,),
    )
    warn = mocker.spy(ah.logger, "warning")
    result = await wait_batch_invoke(
        [
            invoke_actor(Stub.slow_method, tmp_actor1, 0.3),
            invoke_actor(Stub.method_exception, tmp_actor2),
        ]
    )
    assert warn.call_count >= 1
    assert "may cause hang" in warn.call_args[0][0]
    assert result.is_all_successful is False


def test_kill_actors(tmp_actor1, tmp_actor2):
    invoke_actor(Stub.terminate, tmp_actor1).wait()  # Already died

    ah.kill_actors([tmp_actor1, tmp_actor2])
    assert tmp_actor1 not in ah.__actors_cache
    assert tmp_actor2 not in ah.__actors_cache
    assert ray.util.list_named_actors() == []

    ah.kill_actors([tmp_actor1, tmp_actor2])


def test_restart_actor(tmp_actor1):
    assert invoke_actor(Stub.is_restarted, tmp_actor1).wait() is False
    asyncio.run(ah.restart_actors([tmp_actor1]))
    assert invoke_actor(Stub.is_restarted, tmp_actor1).wait() is True

    with pytest.raises(ValueError):
        asyncio.run(ah.restart_actors(["non_existent_actor"]))


def test_actor_call_decorator(tmp_actor1):
    with pytest.raises(ValueError):
        Stub.some_method()
    assert (
        repr(Stub.some_method)
        == "ActorCall(func=Stub.some_method, actor=UNBOUND)"
    )
    assert Stub.some_method.bind(tmp_actor1)() == "ok"

    actor = Stub(tmp_actor1)
    assert actor.some_method() == "ok"
    assert asyncio.run(actor.some_method.async_call()) == "ok"
    assert actor.some_method_alias() == "ok"
    assert actor.some_method_with_arg(1, b="b") == "ok"
    with pytest.raises(AttributeError):
        actor.not_existent_method()


def test_batch_invoke_result():
    actors = ["a", "b"]
    results = ["ok", 999]
    batch = ah.BatchInvokeResult(actors, "foo", results)

    assert batch.is_all_successful
    batch.raise_for_errors()  # no exception expected

    assert batch["a"] == "ok"
    assert batch[0] == "ok"
    assert batch["b"] == 999
    assert batch[1] == 999
    with pytest.raises(KeyError):
        _ = batch["c"]
    with pytest.raises(IndexError):
        _ = batch[2]

    assert batch.all_failed() == []
    assert batch.results == results
    assert batch.as_dict() == {"a": "ok", "b": 999}


def test_batch_invoke_result_with_failure():
    actors = ["a", "b"]
    results = ["ok", Exception("fail")]
    batch = ah.BatchInvokeResult(actors, "foo", results)

    assert not batch.is_all_successful
    with pytest.raises(Exception):
        batch.raise_for_errors()

    assert batch["a"] == "ok"
    assert batch[0] == "ok"
    with pytest.raises(Exception) as exc_info:
        _ = batch[1]
    assert exc_info.value is results[1]

    assert batch.all_failed() == [("b", results[1])]
    with pytest.raises(Exception):
        _ = batch.results
    with pytest.raises(Exception):
        _ = batch.as_dict()


def test_static_stub_invoke(tmp_actor1, tmp_actor2):
    assert invoke_actor(Stub.some_method, tmp_actor1).wait() == "ok"
    assert invoke_actor(Stub.some_method_alias, tmp_actor2).wait() == "ok"
    assert (
        invoke_actor(Stub.some_method_with_arg, tmp_actor1, 1, b="b").wait()
        == "ok"
    )


def test_slow_invoke(tmp_actor1, tmp_actor2):
    # Test that the slow method can be invoked and returns the expected result
    with (
        patch.object(
            ah.ActorBatchInvocation.async_wait,
            "__defaults__",
            (0.1,),
        ),
        patch.object(ah.logger, "info", wraps=ah.logger.info) as mock_log,
    ):
        print(ah.ActorBatchInvocation.async_wait.__defaults__)
        assert asyncio.run(
            invoke_actors(Stub.slow_method, [tmp_actor1, tmp_actor2], 0.2)
        ).results == ["done", "done"]
    assert any(
        "Waiting completing" in call.args[0]
        for call in mock_log.call_args_list
    )

    async def async_cancel_test():
        ref = invoke_actor(Stub.slow_method, tmp_actor1, 1.0)
        task = asyncio.create_task(ref.async_wait())
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return ref.wait()

    assert asyncio.run(async_cancel_test()) == "done"


@pytest.mark.timeout(1, func_only=True)
async def test_wait_ray_node_remove(mocker: MockerFixture):
    ray_node = mocker.patch("ray.nodes")
    ray_node.return_value = [
        {"NodeID": "node1", "Alive": True},
        {"NodeID": "node2", "Alive": True},
    ]

    async def remove_node():
        await asyncio.sleep(0.2)
        ray_node.return_value[1]["Alive"] = False

    bg = asyncio.create_task(remove_node())
    await ah.wait_ray_node_remove(["node2"], interval=0.1)
    await bg


def test_timeout_invoke(tmp_actor1):
    # All timeout raises TimeoutError
    with pytest.raises(TimeoutError):
        invoke_actor(
            Stub.slow_method,
            tmp_actor1,
            1.0,
            _rpc_timeout=0.2,  # type: ignore
        ).wait()
    with pytest.raises(TimeoutError):
        asyncio.run(
            invoke_actor(
                Stub.slow_method,
                tmp_actor1,
                1.0,
                _rpc_timeout=0.2,  # type: ignore
            ).async_wait()
        )
