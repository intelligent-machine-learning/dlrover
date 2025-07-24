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
from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    Callable,
    List,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from .actor_helper import (
    ActorBatchInvocation,
    ActorInvocation,
    get_actor_with_cache,
    invoke_actors,
    invoke_actors_async,
)


@dataclass
class ActorInvocationMeta:
    name: Optional[str] = None
    timeout: Optional[float] = None


META_ATTR_NAME = "__invoke_meta__"
EMPTY_META = ActorInvocationMeta()


def invoke_meta(
    name: Optional[str] = None,
    timeout: Optional[float] = None,
):
    """Decorator to create an ActorInvocationMeta instance."""
    assert timeout is None or timeout > 0, (
        f"Timeout must be a positive number, got {timeout}."
    )

    def decorator(func):
        """Decorator to apply the invoke meta to a function."""
        meta = ActorInvocationMeta(
            name=name or func.__name__,
            timeout=timeout,
        )
        setattr(func, META_ATTR_NAME, meta)

        return func

    return decorator


P = ParamSpec("P")
R = TypeVar("R")


class IActorInfo(Protocol):
    """Common interface for actor information, for extracting name."""

    @property
    def name(self) -> str: ...


def invoke_actor_t(
    func: Callable[P, R], actor_name: str, *args: P.args, **kwargs: P.kwargs
) -> ActorInvocation[R]:
    """Type Safe wrapper for invoking a method on a Ray actor."""
    meta: ActorInvocationMeta = getattr(func, META_ATTR_NAME, EMPTY_META)
    name = meta.name or func.__name__
    return ActorInvocation[R](actor_name, name, *args, **kwargs)


def invoke_actors_t(
    func: Callable[P, R],
    actors: Sequence[Union[str, IActorInfo]],
    *args: P.args,
    **kwargs: P.kwargs,
) -> ActorBatchInvocation[R]:
    """Type Safe wrapper for invoking a method on a Ray actor."""
    meta: ActorInvocationMeta = getattr(func, META_ATTR_NAME, EMPTY_META)
    name = meta.name or func.__name__
    return ActorBatchInvocation[R](
        [
            ActorInvocation[R](
                actor if isinstance(actor, str) else actor.name,
                name,
                *args,
                **kwargs,
            )
            for actor in actors
        ]
    )


T_Stub = TypeVar("T_Stub", covariant=True)


class ActorProxy:
    """A proxy class to interact with Ray actors as if they were local objects."""

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

        @wraps(method)
        def remote_call(*args, **kwargs):
            ref = invoke_actor_t(method, self.actor, *args, **kwargs)
            return ref.wait()

        return remote_call

    @staticmethod
    def wrap(
        actor_name: str, cls: Type[T_Stub], lazy: bool = False
    ) -> "T_Stub":
        """Wraps the actor proxy to return an instance of the class."""
        return ActorProxy(actor_name, cls, warmup=not lazy)  # type: ignore


class BatchActorProxy:
    """
    A proxy class to interact with multiple Ray actors as if they were local objects.

    All return values in Stub should be BatchInvokeResult.
    """

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
        meta: ActorInvocationMeta = getattr(method, META_ATTR_NAME, EMPTY_META)

        if asyncio.iscoroutinefunction(method):
            return partial(invoke_actors_async, self.actors, meta.name or name)
        else:
            return partial(invoke_actors, self.actors, meta.name or name)

    @staticmethod
    def wrap(actor_name: List[str], cls: Type[T_Stub]) -> "T_Stub":
        """Wraps the actor proxy to return an instance of the class."""
        return BatchActorProxy(actor_name, cls)  # type: ignore[return-value]
