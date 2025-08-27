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
import abc
import asyncio
from abc import ABC
from functools import cached_property
from typing import (
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import ray
from ray.actor import ActorClass, ActorHandle
from ray.exceptions import (
    ActorUnavailableError,
    GetTimeoutError,
    RayActorError,
    RayTaskError,
)

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.common.constant import RAY_HANG_CHECK_INTERVAL

__actors_cache: Dict[str, ActorHandle] = {}
T = TypeVar("T")


def reset_actor_cache(name: str):
    __actors_cache.pop(name, None)


def refresh_actor_cache(name: str):
    """Refresh the cache for a Ray actor by its name."""
    try:
        reset_actor_cache(name)
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


class InvocationRef(Generic[T], ABC):
    @property
    @abc.abstractmethod
    def pending(self) -> bool:  # pragma: no-cover
        """Check if the invocation is still pending."""
        ...

    @property
    @abc.abstractmethod
    def result(self) -> T:  # pragma: no-cover
        """Get the result of the invocation, raising an exception if it failed."""
        ...

    @abc.abstractmethod
    def wait(self) -> T:  # pragma: no-cover
        """Wait for the result of the invocation, blocking until it is ready."""
        ...

    @abc.abstractmethod
    async def async_wait(self) -> T:  # pragma: no-cover
        """Wait for the result of the invocation asynchronously."""
        ...

    def __await__(self) -> Generator[None, None, T]:
        """Await the result of the invocation."""
        return self.async_wait().__await__()


class ActorInvocation(InvocationRef[T]):
    """A class to represent an invocation of a method on a Ray actor.
    wraps ObjectRef and handles retries for actor errors.
    """

    def __init__(
        self,
        actor_name: str,
        method_name: str,
        *args,
        **kwargs,
    ):
        self.actor_name = actor_name
        self.method_name = method_name
        self.display_name: str = kwargs.pop("_display_name", method_name)
        self.args = args
        self.kwargs = kwargs

        self._result: Union[T, ray.ObjectRef, Exception] = None  # type: ignore
        self._retry_count = 3

        self._invoke()

    def _invoke(self):
        # Return ObjectRef or Exception that occurs during invocation.
        try:
            actor = get_actor_with_cache(self.actor_name)
            self._result = (
                getattr(actor, self.method_name)
                .options(name=self.display_name)
                .remote(*self.args, **self.kwargs)
            )
        except ValueError as e:
            # already logged by get_actor_with_cache
            # logger.error(f"Actor {self.actor_name} not found: {e}")
            self._result = e
        except AttributeError as e:
            logger.error(
                f"Method {self.method_name} not found on actor {self.actor_name}: {e}"
            )
            self._result = e
        except Exception as e:
            logger.error(
                f"Fail to submit task {self.display_name} to {self.actor_name}: {e}"
            )
            self._result = e
        self._check_result()

    def resolve_sync(self, timeout: Optional[float] = None):
        """Wait for the result of the invocation asynchronously.
        Note: May still be pending after this call if retries are in progress.
        """
        assert isinstance(self._result, ray.ObjectRef)
        try:
            self._result = cast(T, ray.get(self._result, timeout=timeout))
        except GetTimeoutError:
            return  # still pending, will retry later
        except RayActorError as e:
            self._result = e  # known error,handle in _check_result
        except RayTaskError as e:
            self._result = RuntimeError(
                f"Error executing {self.display_name} on {self.actor_name}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected exception executing {self.display_name} on {self.actor_name}: {e}"
            )
            self._result = e
        self._check_result()

    async def resolve_async(self, timeout: Optional[float] = None):
        """Wait for the result of the invocation asynchronously.
        Note: May still be pending after this call if retries are in progress.
        """
        assert isinstance(self._result, ray.ObjectRef)

        try:
            co = asyncio.shield(self._result)
            if timeout is not None:
                co = asyncio.wait_for(co, timeout=timeout)  # type: ignore[assignment]
            self._result = cast(T, await co)
        except TimeoutError:
            return
        except RayActorError as e:
            self._result = e  # known error,handle in _check_result
        except RayTaskError as e:
            self._result = RuntimeError(
                f"Error executing {self.display_name} on {self.actor_name}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected exception executing {self.display_name} on {self.actor_name}: {type(e)}"
            )
            self._result = e
        self._check_result()

    def _check_result(self):
        """Check if the invocation should be retried."""
        # retry for actor restart or unavailable errors
        if self._retry_count > 0 and isinstance(self._result, RayActorError):
            if (
                self.method_name == "__ray_terminate__"
                or self.method_name == "shutdown"
            ):
                self._result = cast(T, None)  # Success for shutdown
                return

            self._retry_count -= 1
            logger.warning(
                f"ActorError when executing {self.display_name} on {self.actor_name},"
                f" retrying ({self._retry_count} retries left); {self._result}"
            )
            reset_actor_cache(self.actor_name)
            self._invoke()

    @property
    def pending(self) -> bool:
        """Check if the invocation is still pending."""
        return isinstance(self._result, ray.ObjectRef)

    def wait(self) -> T:
        """Wait for the result of the invocation, blocking until it is ready."""
        while self.pending:
            self.resolve_sync()
        return self.result

    async def async_wait(self) -> T:
        """Wait for the result of the invocation asynchronously."""
        while self.pending:
            await self.resolve_async()
        return self.result

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


class ActorBatchInvocation(Generic[T], InvocationRef["BatchInvokeResult[T]"]):
    """A class to represent an invocation of a method on multiple Ray actors."""

    def __init__(self, refs: List[ActorInvocation[T]]):
        assert len(refs) > 0, "At least one actor must be specified."
        self.refs = refs
        self.display_name = refs[0].display_name

    @property
    def pending(self) -> bool:
        """Check if any invocation is still pending."""
        return any(ref.pending for ref in self.refs)

    def _resolve_sync(self, timeout: Optional[float]):
        """Wait for all invocations to complete."""
        waiting_refs: Dict[ray.ObjectRef, ActorInvocation[T]] = {
            ref._result: ref
            for ref in self.refs
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
        if not_ready:
            stragglers = [ref.actor_name for ref in not_ready]
            logger.info(
                f"Waiting completing {self.display_name} ...: {stragglers}"
            )

    def wait(
        self, straggler_timeout: float = RAY_HANG_CHECK_INTERVAL
    ) -> "BatchInvokeResult[T]":
        """Wait for all invocations to complete, blocking until they are ready."""
        while self.pending:
            self._resolve_sync(timeout=straggler_timeout)
        return self.result

    async def async_wait(
        self, monitor_interval: float = RAY_HANG_CHECK_INTERVAL
    ):
        async def wait(ref: ActorInvocation[T]):
            """Resolve the invocation and return the result."""
            while ref.pending:
                await ref.resolve_async()

        async def monitor():
            """Monitor the invocations and log progress."""
            while True:
                await asyncio.sleep(delay=monitor_interval)
                stragglers = [
                    ref.actor_name for ref in self.refs if ref.pending
                ]
                logger.info(
                    f"Waiting completing {self.display_name} ...: {stragglers}"
                )

        monitor_task = asyncio.create_task(monitor())
        await asyncio.gather(*[wait(ref) for ref in self.refs])
        monitor_task.cancel()
        return self.result

    @property
    def result(self) -> "BatchInvokeResult[T]":
        """Get the results of all invocations."""
        assert not self.pending, (
            "Invocations are still pending. Call resolve first."
        )
        results = [ref.result_or_exception for ref in self.refs]
        return BatchInvokeResult(
            [ref.actor_name for ref in self.refs],
            self.display_name,
            results,
        )


def kill_actors(actors: List[str]):
    """Kill Ray actors by their names."""
    logger.info(f"Killing actors: {actors}")
    refs = []
    for actor in actors:
        try:
            actor_handle = get_actor_with_cache(actor)
            refs.append(actor_handle.__ray_terminate__.remote())
        except ValueError:
            # Actor not found, continue
            continue
    ray.wait(
        refs, num_returns=len(refs), timeout=10.0
    )  # try graceful shutdown
    for actor in actors:
        try:
            get_actor_with_cache(actor)
            ray.kill(actor, no_restart=True)
        except ValueError:
            # Actor not found, continue
            continue
    for node in actors:
        __actors_cache.pop(node, None)


async def restart_actors(actors: List[str]):
    """Restart Ray actors by their names."""
    logger.info(f"Restarting actors: {actors}")
    for actor in actors:
        try:
            ray.kill(get_actor_with_cache(actor), no_restart=False)
            refresh_actor_cache(actor)
        except ValueError:
            raise ValueError(f"Actor {actor} not found for restart.")
    await wait_ready(actors.copy())
    logger.info(f"Actors restarted: {actors}")


async def wait_ready(actors: List[str]):
    """Wait for all actors to be ready."""

    async def wait_one(actor_name: str):
        """Handle the actor readiness."""
        try:
            # Not use cache here
            actor = ray.get_actor(actor_name)  # must keep ref
            await actor.__ray_ready__.remote()
            actors.remove(actor_name)
        except ActorUnavailableError:
            return  # expected for restarting

    while actors:
        logger.info(f"Waiting for actors to be ready: {actors}")
        await asyncio.gather(
            *[wait_one(actor_name) for actor_name in actors.copy()]
        )
        if actors:
            await asyncio.sleep(RAY_HANG_CHECK_INTERVAL)


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
        Raise Exception if the result is an error.
        """
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

    def log_errors(self):
        """Log errors for all failed actors."""
        for actor, exc in self.all_failed():
            logger.error(
                f"Actor {actor} failed executing {self.method}", exc_info=exc
            )

    def raise_for_errors(self):
        """Raise an exception if any actor failed."""
        if self.is_all_successful:
            return
        self.log_errors()
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
