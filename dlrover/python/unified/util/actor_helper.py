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
from functools import cached_property, partial
from typing import (
    Dict,
    Generic,
    List,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import ray
from ray.actor import ActorClass, ActorHandle
from ray.exceptions import RayActorError

from dlrover.python.common.log import default_logger as logger

__actors_cache: Dict[str, ActorHandle] = {}
T = TypeVar("T")


def refresh_actor_cache(name: str):
    try:
        __actors_cache.pop(name, None)
        __actors_cache[name] = ray.get_actor(name)
    except ValueError:
        logger.error(f"Actor {name} not found")
        raise


def get_actor_with_cache(name: str):
    """Get a Ray actor by its name, optionally refreshing the cache."""
    # It's safe without a lock, as actors are identified by their names,
    if name not in __actors_cache:
        refresh_actor_cache(name)
    return __actors_cache[name]


def as_actor_class(cls: type) -> ActorClass:
    """Convert a class to a Ray actor class if it is not already."""
    if isinstance(cls, ActorClass):
        return cls
    return ray.remote(cls)  # type: ignore[return-value]


def __invoke_actor(
    actor_name: str, method_name: str, *args, **kwargs
) -> ray.ObjectRef:
    """Invoke a method on a Ray actor."""
    try:
        actor = get_actor_with_cache(actor_name)
        return getattr(actor, method_name).remote(*args, **kwargs)
    except AttributeError as e:
        logger.error(
            f"Method {method_name} not found on actor {actor_name}: {e}"
        )
        return ray.put(None)


def invoke_actors(actors: List[str], method_name: str, *args, **kwargs):
    """Execute a method on all nodes."""
    tasks = [
        __invoke_actor(name, method_name, *args, **kwargs) for name in actors
    ]

    results: list = [None] * len(actors)
    waiting = {task: task_id for task_id, task in enumerate(tasks)}
    while len(waiting) > 0:
        ready, not_ready = ray.wait(
            list(waiting.keys()), num_returns=len(waiting), timeout=10
        )
        next_waiting = {task: waiting[task] for task in not_ready}
        for task in ready:
            task_id = waiting.pop(task)
            actor = actors[task_id]
            try:
                result = ray.get(task)
                results[task_id] = result
            except RayActorError as e:
                if method_name == "shutdown":
                    results[task_id] = None  # Expected for shutdown
                    continue
                logger.error(f"Error executing {method_name} on {actor}: {e}")
                refresh_actor_cache(actor)
                task = __invoke_actor(actor, method_name, *args, **kwargs)
                next_waiting[task] = task_id
            except Exception as e:
                logger.error(
                    f"Unexpected error executing {method_name} on {actor}: {e}"
                )
                results[task_id] = e
        waiting = next_waiting
        if len(waiting) > 0:
            stragglers = [actors[task_id] for task_id in waiting.values()]
            logger.info(
                f"Waiting for {len(stragglers)} tasks ({stragglers}) to "
                f"complete {method_name} ..."
            )
    return BatchInvokeResult(actors, method_name, results)


def invoke_actor(actor_name: str, method_name: str, *args, **kwargs):
    """Call a method on a Ray actor by its name."""
    return invoke_actors([actor_name], method_name, *args, **kwargs)[0]


def kill_actors(actors: List[str]):
    """Kill Ray actors by their names."""
    logger.info(f"Killing actors: {actors}")
    toKill = actors.copy()
    while len(toKill) > 0:
        name = toKill.pop()
        try:
            ray.wait([__invoke_actor(name, "shutdown")])
            actor = get_actor_with_cache(name)
            ray.kill(actor, no_restart=True)
        except ValueError:
            # Actor not found, continue
            continue
        except RayActorError:
            refresh_actor_cache(name)
            toKill.append(name)

    for node in actors:
        __actors_cache.pop(node, None)


async def invoke_actor_async(
    actor_name: str, method_name: str, *args, **kwargs
) -> Union[T, Exception]:
    """Call a method on a Ray actor by its name."""
    while True:
        try:
            res = __invoke_actor(actor_name, method_name, *args, **kwargs)
            return await res
        except RayActorError as e:
            if method_name == "shutdown":
                return cast(T, None)  # Success for shutdown
            print(f"Error executing {method_name} on {actor_name}: {e}")
            refresh_actor_cache(actor_name)
            continue  # Retry with the refreshed actor
        except Exception as e:
            print(
                f"Unexpected error executing {method_name} on {actor_name}: "
                f"{e}"
            )
            return e


async def invoke_actors_async(
    actors: List[str], method_name: str, *args, **kwargs
) -> "BatchInvokeResult[T]":
    res: List[Union[T, Exception]] = await asyncio.gather(
        *[
            invoke_actor_async(actor, method_name, *args, **kwargs)
            for actor in actors
        ]
    )
    return BatchInvokeResult[T](actors, method_name, res)


class ActorProxy:
    def __init__(self, actor: str, stub_cls: type, warmup: bool = True):
        self.actor = actor
        self.stub_cls = stub_cls
        if warmup:
            get_actor_with_cache(actor)  # warmup actor

    def __getattr__(self, name):
        method = getattr(self.stub_cls, name, None)
        if method is None:
            raise AttributeError(
                f"Method {name} not found in actor {self.actor}."
            )
        if asyncio.iscoroutinefunction(method):
            return partial(invoke_actor_async, self.actor, name)
        else:
            return partial(invoke_actor, self.actor, name)

    @staticmethod
    def wrap(actor_name: str, cls: Type[T], lazy: bool = False) -> "T":
        """Wraps the actor proxy to return an instance of the class."""
        return ActorProxy(actor_name, cls, warmup=not lazy)  # type: ignore


class BatchInvokeResult(Generic[T]):
    def __init__(
        self,
        actors: List[str],
        method_name: str,
        results: List[Union[T, Exception]],
    ):
        self.actors = actors
        self.method = method_name
        self._results = results

    @overload
    def __getitem__(self, item: int, /) -> T:
        ...

    @overload
    def __getitem__(self, actor: str, /) -> T:
        ...

    def __getitem__(self, item: Union[str, int]) -> T:
        """Get the result for a specific actor by index or name.
        Raise Exception if the result is an error."""
        if isinstance(item, str):
            index = self.actors.index(item)
            if index == -1:
                raise KeyError(f"Actor {item} not found in results.")
            item = index
        if item < 0 or item >= len(self._results):
            raise IndexError("Index out of range.")
        res = self._results[item]
        if isinstance(res, Exception):
            raise res
        return res

    @cached_property
    def is_all_successful(self) -> bool:
        """Check if all results are successful."""
        return all(
            not isinstance(result, Exception) for result in self._results
        )

    def all_failed(self) -> List[Tuple[str, Exception]]:
        """Get a list of failed actors with their exceptions."""
        return [
            (actor, result)
            for actor, result in zip(self.actors, self._results)
            if isinstance(result, Exception)
        ]

    def raise_for_errors(self):
        """Raise an exception if any actor failed."""
        if self.is_all_successful:
            return

        for actor, exc in self.all_failed():
            logger.error(
                f"Actor {actor} failed executing {self.method}: {exc}"
            )
        raise Exception(
            f"Some actors failed executing {self.method}. "
            f"Failed actors: "
            f"{', '.join(actor for actor, _ in self.all_failed())}"
        )

    @property
    def results(self) -> List[T]:
        """Return results as a list of successful results."""
        self.raise_for_errors()
        return [cast(T, result) for result in self._results]

    def as_dict(self) -> Dict[str, T]:
        """Return results as a dictionary mapping actor names to results."""
        return {
            actor: result for actor, result in zip(self.actors, self.results)
        }


class BatchActorProxy:
    def __init__(self, actors: List[str], stub_cls: type):
        self.actors = actors
        # Optionally filter methods if a class is provided
        self.stub_cls = stub_cls

    def __getattr__(self, name):
        method = getattr(self.stub_cls, name, None)
        if method is None:
            raise AttributeError(
                f"Method {name} not found in class {self.stub_cls.__name__}."
            )
        # if async
        if asyncio.iscoroutinefunction(method):
            return partial(invoke_actors_async, self.actors, name)
        else:
            return partial(invoke_actors, self.actors, name)

    @staticmethod
    def wrap(actor_name: List[str], cls: Type[T]) -> "T":
        """Wraps the actor proxy to return an instance of the class."""
        return BatchActorProxy(actor_name, cls)  # type: ignore[return-value]
