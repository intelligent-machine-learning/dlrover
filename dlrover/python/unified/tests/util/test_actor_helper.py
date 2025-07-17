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
from typing import Any, Protocol
from unittest.mock import patch

import pytest
import ray
from ray.actor import ActorClass

import dlrover.python.unified.util.actor_helper as ah
from dlrover.python.unified.util.actor_proxy import (
    ActorProxy,
    BatchActorProxy,
    invoke_meta,
)


@ray.remote
class SimpleActor:
    def some_method(self):
        return "ok"

    def method_exception(self):
        raise Exception("This is a test exception")

    def method_relaunch(self):
        if not ray.get_runtime_context().was_current_actor_reconstructed:
            ray.kill(ray.get_runtime_context().current_actor, no_restart=False)
        return "restarted"


class SimpleActorStub(Protocol):
    def some_method(self): ...

    @invoke_meta(name="some_method")
    async def some_method_async(self): ...


class SimpleActorBatchStub(Protocol):
    def some_method(self) -> ah.BatchInvokeResult[str]: ...

    @invoke_meta(name="some_method")
    async def some_method_async(self) -> ah.BatchInvokeResult[str]: ...


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
    assert ah.invoke_actor(tmp_actor1, "some_method") == "ok"

    with pytest.raises(AttributeError):
        ah.invoke_actor(tmp_actor1, "non_existent_method")

    assert ah.invoke_actor(tmp_actor1, "method_relaunch") == "restarted"
    with pytest.raises(Exception, match="This is a test exception"):
        ah.invoke_actor(tmp_actor1, "method_exception")

    assert (
        ah.invoke_actor(tmp_actor1, "__ray_terminate__") is None
    )  # No ActorDieError when shutdown


def test_invoke_actor_async(tmp_actor1):
    assert (
        asyncio.run(ah.invoke_actor_async(tmp_actor1, "some_method")) == "ok"
    )
    with pytest.raises(AttributeError):
        assert asyncio.run(
            ah.invoke_actor_async(tmp_actor1, "non_existent_method")
        )

    assert (
        asyncio.run(ah.invoke_actor_async(tmp_actor1, "method_relaunch"))
        == "restarted"
    )
    with pytest.raises(Exception, match="This is a test exception"):
        assert asyncio.run(
            ah.invoke_actor_async(tmp_actor1, "method_exception")
        )


def test_invoke_actors(tmp_actor1, tmp_actor2):
    result = ah.invoke_actors([tmp_actor1, tmp_actor2], "some_method")
    assert result.is_all_successful
    assert result[0] == "ok"
    assert result[1] == "ok"
    result2 = asyncio.run(
        ah.invoke_actors_async([tmp_actor1, tmp_actor2], "some_method")
    )
    assert result.results == result2.results


def test_kill_actors(tmp_actor1, tmp_actor2):
    ah.invoke_actor(tmp_actor2, "__ray_terminate__")  # Already died

    ah.kill_actors([tmp_actor1, tmp_actor2])
    assert tmp_actor1 not in ah.__actors_cache
    assert tmp_actor2 not in ah.__actors_cache
    assert ray.util.list_named_actors() == []

    ah.kill_actors([tmp_actor1, tmp_actor2])


def test_actor_proxy(tmp_actor1):
    actor = ActorProxy.wrap(tmp_actor1, SimpleActorStub)
    assert actor.some_method() == "ok"
    with pytest.raises(AttributeError):
        actor.non_existent_method()  # type: ignore[union-attr]
    assert asyncio.run(actor.some_method_async()) == "ok"


def test_batch_actor_proxy(tmp_actor1, tmp_actor2):
    actors = BatchActorProxy.wrap(
        [tmp_actor1, tmp_actor2], SimpleActorBatchStub
    )
    result = actors.some_method()
    assert result.is_all_successful
    assert result.results == ["ok", "ok"]

    result_async = asyncio.run(actors.some_method_async())
    assert result_async.is_all_successful
    assert result_async.results == ["ok", "ok"]

    with pytest.raises(AttributeError):
        actors.non_existent_method()  # type: ignore[union-attr]


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
