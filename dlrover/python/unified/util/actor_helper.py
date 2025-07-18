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
from functools import cached_property
from typing import (
    Dict,
    Generator,
    Generic,
    List,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import ray
from ray.actor import ActorClass, ActorHandle
from ray.exceptions import ActorUnavailableError, RayActorError, RayTaskError

from dlrover.python.common.log import default_logger as logger

__actors_cache: Dict[str, ActorHandle] = {}
T = TypeVar("T")


def refresh_actor_cache(name: str):
    """Refresh the cache for a Ray actor by its name."""
    try:
        __actors_cache.pop(name, None)
        __actors_cache[name] = ray.get_actor(name)
    except ValueError:
        logger.error(f"Actor {name} not found")
        raise


def get_actor_with_cache(name: str):
    """Get a Ray actor by its name."""
    # It's safe without a lock, as actors are identified by their names,
    if name not in __actors_cache:
        refresh_actor_cache(name)
    return __actors_cache[name]


def as_actor_class(cls: type) -> ActorClass:
    """Convert a class to a Ray actor class if it is not already."""
    if isinstance(cls, ActorClass):
        return cls
    return ray.remote(cls)  # type: ignore[return-value]


class ActorInvocation(Generic[T]):
    """A class to represent an invocation of a method on a Ray actor.
    wraps ObjectRef and handles retries for actor errors.
    """

    def __init__(self, actor_name: str, method_name: str, *args, **kwargs):
        self.actor_name = actor_name
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs

        self._result: Union[T, ray.ObjectRef, Exception] = None  # type: ignore
        self._retry_count = 3

        self._invoke()

    def _invoke(self):
        # Return ObjectRef or Exception that occurs during invocation.
        try:
            actor = get_actor_with_cache(self.actor_name)
            self._result = getattr(actor, self.method_name).remote(
                *self.args, **self.kwargs
            )
        except AttributeError as e:
            logger.error(
                f"Method {self.method_name} not found on actor {self.actor_name}: {e}"
            )
            self._result = e
        except Exception as e:
            logger.error(
                f"Fail to submit task {self.method_name} to {self.actor_name}: {e}"
            )
            self._result = e
        self._check_result()

    def resolve_sync(self):
        """Wait for the result of the invocation."""
        assert isinstance(self._result, ray.ObjectRef)
        try:
            self._result = cast(T, ray.get(self._result))
        except RayTaskError as e:
            self._result = RuntimeError(
                f"Error executing {self.method_name} on {self.actor_name}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected exception executing {self.method_name} on {self.actor_name}: {e}"
            )
            self._result = e
        self._check_result()

    async def resolve_async(self):
        """Wait for the result of the invocation."""
        assert isinstance(self._result, ray.ObjectRef)

        try:
            self._result = cast(T, await self._result)
        except RayTaskError as e:
            self._result = RuntimeError(
                f"Error executing {self.method_name} on {self.actor_name}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected exception executing {self.method_name} on {self.actor_name}: {e}"
            )
            self._result = e
        self._check_result()

    def _check_result(self):
        """Check if the invocation should be retried."""
        # retry for actor restart or unavailable errors
        if self._retry_count > 0 and isinstance(
            self._result, (RayActorError, ActorUnavailableError)
        ):
            if (
                self.method_name == "__ray_terminate__"
                or self.method_name == "shutdown"
            ):
                self._result = cast(T, None)  # Success for shutdown
                return

            self._retry_count -= 1
            logger.warning(
                f"ActorError when executing {self.method_name} on {self.actor_name},"
                f" retrying ({self._retry_count} retries left); {self._result}"
            )
            refresh_actor_cache(self.actor_name)
            self._invoke()

    @property
    def pending(self) -> bool:
        """Check if the invocation is still pending."""
        return isinstance(self._result, ray.ObjectRef)

    @property
    def result_or_exception(self) -> Union[T, Exception]:
        """Get the result of the invocation."""
        assert not self.pending, (
            "Invocation is still pending. Call resolve first."
        )
        assert not isinstance(self._result, ray.ObjectRef)
        return self._result

    @property
    def result(self) -> T:
        """Get the result of the invocation, raising an exception if it failed."""
        res = self.result_or_exception
        if isinstance(res, Exception):
            raise res
        return cast(T, res)

    def wait(self) -> T:
        """Wait for the result of the invocation, blocking until it is ready."""
        while self.pending:
            self.resolve_sync()
        return self.result

    def __await__(self) -> Generator[None, None, T]:
        """Await the result of the invocation."""
        while isinstance(self._result, ray.ObjectRef):
            yield from self.resolve_async().__await__()
        return self.result

    @staticmethod
    def wait_many(
        refs: List["ActorInvocation[T]"], timeout: float = 10.0
    ) -> List["ActorInvocation[T]"]:
        """Wait for all invocations to complete, return not ready refs."""
        waiting_refs: Dict[ray.ObjectRef, ActorInvocation[T]] = {
            ref._result: ref
            for ref in refs
            if isinstance(ref._result, ray.ObjectRef)
        }
        if len(waiting_refs) > 0:
            ready, _ = ray.wait(
                list(waiting_refs.keys()),
                timeout=timeout,
                num_returns=len(waiting_refs),
            )
            for task in ready:
                ref = waiting_refs.pop(task)
                ref.resolve_sync()
                if isinstance(ref._result, ray.ObjectRef):
                    waiting_refs[ref._result] = ref  # re-add if still pending
        not_ready = list(waiting_refs.values())
        return not_ready


class ActorBatchInvocation(Generic[T]):
    """A class to represent an invocation of a method on multiple Ray actors."""

    def __init__(self, refs: List[ActorInvocation[T]]):
        assert len(refs) > 0, "At least one actor must be specified."
        self.refs = refs
        self.method_name = refs[0].method_name

    @property
    def pending(self) -> bool:
        """Check if any invocation is still pending."""
        return any(ref.pending for ref in self.refs)

    def resolve_sync(self):
        """Wait for all invocations to complete."""
        waiting = self.refs
        while len(waiting) > 0:
            waiting = ActorInvocation.wait_many(waiting, timeout=10.0)
            if len(waiting) > 0:
                stragglers = [ref.actor_name for ref in waiting]
                logger.info(
                    f"Waiting completing {self.method_name} ...: {stragglers}"
                )

    async def resolve_async(self):
        async def wait(ref: ActorInvocation[T]):
            """Resolve the invocation and return the result."""
            while ref.pending:
                await ref.resolve_async()

        await asyncio.gather(*[wait(ref) for ref in self.refs])

    def result(self) -> "BatchInvokeResult[T]":
        """Get the results of all invocations."""
        assert not self.pending, (
            "Invocations are still pending. Call resolve first."
        )
        results = [ref.result_or_exception for ref in self.refs]
        return BatchInvokeResult(
            [ref.actor_name for ref in self.refs],
            self.method_name,
            results,
        )

    def wait(self) -> "BatchInvokeResult[T]":
        """Wait for all invocations to complete and return the results."""
        self.resolve_sync()
        return self.result()

    def __await__(self) -> Generator[None, None, "BatchInvokeResult[T]"]:
        """Await the results of all invocations."""
        yield from self.resolve_async().__await__()
        return self.result()


def invoke_actors(
    actors: List[str], method_name: str, *args, **kwargs
) -> "BatchInvokeResult[T]":
    """Execute a method on all nodes."""
    ref = ActorBatchInvocation(
        [
            ActorInvocation[T](actor, method_name, *args, **kwargs)
            for actor in actors
        ]
    )
    return ref.wait()


def invoke_actor(actor_name: str, method_name: str, *args, **kwargs) -> T:  # type: ignore[type-var]
    """Call a method on a Ray actor by its name."""
    ref = ActorInvocation[T](actor_name, method_name, *args, **kwargs)
    return ref.wait()


def kill_actors(actors: List[str]):
    """Kill Ray actors by their names."""
    logger.info(f"Killing actors: {actors}")
    toKill = actors.copy()
    while len(toKill) > 0:
        name = toKill.pop()
        try:
            invoke_actor(name, "__ray_terminate__")
            actor = get_actor_with_cache(name)
            ray.kill(actor, no_restart=True)
        except ValueError:
            # Actor not found, continue
            continue

    for node in actors:
        __actors_cache.pop(node, None)


async def invoke_actor_async(
    actor_name: str,
    method_name: str,
    *args,
    **kwargs,
) -> T:
    """Call a method on a Ray actor by its name."""
    ref = ActorInvocation[T](actor_name, method_name, *args, **kwargs)
    return await ref


async def invoke_actors_async(
    actors: List[str], method_name: str, *args, **kwargs
) -> "BatchInvokeResult[T]":
    """Execute a method on all nodes asynchronously."""

    ref = ActorBatchInvocation(
        [
            ActorInvocation[T](actor, method_name, *args, **kwargs)
            for actor in actors
        ]
    )
    return await ref


class BatchInvokeResult(Generic[T]):
    """A class to hold results from invoking methods on multiple actors."""

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
    def __getitem__(self, item: int, /) -> T: ...

    @overload
    def __getitem__(self, actor: str, /) -> T: ...

    def __getitem__(self, item: Union[str, int]) -> T:
        """Get the result for a specific actor by index or name.
        Raise Exception if the result is an error."""
        if isinstance(item, str):
            try:
                index = self.actors.index(item)
            except ValueError:
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
                f"Actor {actor} failed executing {self.method}", exc_info=exc
            )
        raise Exception(
            f"Some actors failed executing {self.method}: "
            f"{[actor for actor, _ in self.all_failed()]}"
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
