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
from ray.actor import ActorClass

import dlrover.python.unified.util.actor_helper as ah
from dlrover.python.unified.util.actor_proxy import (
    ActorProxy,
    invoke_actor_t,
    invoke_actors_t,
    invoke_meta,
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

    def method_relaunch(self):
        if not ray.get_runtime_context().was_current_actor_reconstructed:
            ray.kill(ray.get_runtime_context().current_actor, no_restart=False)
        return "restarted"

    def is_restarted(self):
        return ray.get_runtime_context().was_current_actor_reconstructed


class Stub(ActorProxy):
    @staticmethod
    def some_method() -> str: ...  # type: ignore[empty-body]
    @staticmethod
    def some_method_with_arg(a: int, b: str) -> str: ...  # type: ignore[empty-body]
    @staticmethod
    def method_relaunch() -> str: ...  # type: ignore[empty-body]
    @staticmethod
    def method_exception() -> str: ...  # type: ignore[empty-body]
    @staticmethod
    def is_restarted() -> bool: ...  # type: ignore[empty-body]

    @staticmethod
    def not_existent_method() -> str: ...  # type: ignore[empty-body]
    @staticmethod
    @invoke_meta(name="some_method")
    def some_method_alias() -> str: ...  # type: ignore[empty-body]
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
        invoke_actor_t(Stub.some_method, "non_existent_actor").wait()
    assert invoke_actor_t(Stub.some_method, tmp_actor1).wait() == "ok"
    assert invoke_actor_t(Stub.some_method_alias, tmp_actor1).wait() == "ok"
    assert (
        invoke_actor_t(Stub.some_method_with_arg, tmp_actor1, 1, b="b").wait()
        == "ok"
    )

    with pytest.raises(AttributeError):
        invoke_actor_t(Stub.not_existent_method, tmp_actor1).wait()
    assert (
        invoke_actor_t(Stub.method_relaunch, tmp_actor1).wait() == "restarted"
    )
    with pytest.raises(Exception, match="This is a test exception"):
        invoke_actor_t(Stub.method_exception, tmp_actor1).wait()

    # No ActorDieError when shutdown
    assert invoke_actor_t(Stub.terminate, tmp_actor1).wait() is None


def test_invoke_actor_async(tmp_actor1):
    def async_wait(invoke: ah.ActorInvocation):
        return asyncio.run(invoke.async_wait())

    with pytest.raises(ValueError):
        async_wait(invoke_actor_t(Stub.some_method, "non_existent_actor"))
    assert async_wait(invoke_actor_t(Stub.some_method, tmp_actor1)) == "ok"
    with pytest.raises(AttributeError):
        async_wait(invoke_actor_t(Stub.not_existent_method, tmp_actor1))

    assert (
        async_wait(invoke_actor_t(Stub.method_relaunch, tmp_actor1))
        == "restarted"
    )
    with pytest.raises(Exception, match="This is a test exception"):
        async_wait(invoke_actor_t(Stub.method_exception, tmp_actor1))


def test_invoke_actors(tmp_actor1, tmp_actor2):
    result = invoke_actors_t(Stub.some_method, [tmp_actor1, tmp_actor2]).wait()
    assert result.is_all_successful
    assert result[0] == "ok"
    assert result[1] == "ok"
    result2 = asyncio.run(
        invoke_actors_t(
            Stub.some_method, [tmp_actor1, tmp_actor2]
        ).async_wait()
    )
    assert result.results == result2.results


def test_kill_actors(tmp_actor1, tmp_actor2):
    invoke_actor_t(Stub.terminate, tmp_actor1).wait()  # Already died

    ah.kill_actors([tmp_actor1, tmp_actor2])
    assert tmp_actor1 not in ah.__actors_cache
    assert tmp_actor2 not in ah.__actors_cache
    assert ray.util.list_named_actors() == []

    ah.kill_actors([tmp_actor1, tmp_actor2])


def test_restart_actor(tmp_actor1):
    assert invoke_actor_t(Stub.is_restarted, tmp_actor1).wait() is False
    asyncio.run(ah.restart_actors([tmp_actor1]))
    assert invoke_actor_t(Stub.is_restarted, tmp_actor1).wait() is True

    with pytest.raises(ValueError):
        asyncio.run(ah.restart_actors(["non_existent_actor"]))


def test_actor_proxy(tmp_actor1):
    actor = Stub.bind(tmp_actor1)
    assert actor.some_method() == "ok"
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
    assert invoke_actor_t(Stub.some_method, tmp_actor1).wait() == "ok"
    assert invoke_actor_t(Stub.some_method_alias, tmp_actor2).wait() == "ok"
    assert (
        invoke_actor_t(Stub.some_method_with_arg, tmp_actor1, 1, b="b").wait()
        == "ok"
    )

    assert invoke_actors_t(
        Stub.some_method, [tmp_actor1, tmp_actor2]
    ).wait().results == [
        "ok",
        "ok",
    ]
