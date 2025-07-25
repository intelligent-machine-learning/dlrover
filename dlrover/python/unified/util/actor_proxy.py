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

from dataclasses import dataclass
from functools import wraps
from typing import (
    Any,
    Callable,
    ClassVar,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

from .actor_helper import (
    ActorBatchInvocation,
    ActorInvocation,
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


def _proxy_wrapper(method: Callable[..., Any], actor_name: Optional[str]):
    method = getattr(method, "__origin__", method)

    @wraps(method)
    def remote_call(*args, **kwargs):
        if not actor_name:
            raise TypeError(
                "ACTOR_NAME not defined, you must use bind(actor_name)."
            )
        ref = invoke_actor_t(method, actor_name, *args, **kwargs)
        return ref.wait()

    setattr(remote_call, "__origin__", method)

    return remote_call


T = TypeVar("T", bound="type")


class ActorProxy:
    """A proxy class to interact with Ray actors as if they were local objects."""

    ACTOR_NAME: ClassVar[str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        actor_name = getattr(cls, "ACTOR_NAME", None)
        cls2: Optional[type] = cls
        while cls2 is not None and cls2 != ActorProxy:
            for name, method in cls2.__dict__.items():
                if not callable(method) or name.startswith("__"):
                    continue
                if not isinstance(method, staticmethod):
                    raise TypeError(
                        f"Method {method} must be a static method."
                    )
                setattr(
                    cls,
                    name,
                    staticmethod(_proxy_wrapper(method.__func__, actor_name)),
                )
            cls2 = cls2.__base__

    @classmethod
    def bind(cls: T, actor_name: str) -> T:
        """Bind the actor name to the class."""

        class BoundActorProxy(cls):  # type: ignore[misc,valid-type]
            ACTOR_NAME = actor_name

        return BoundActorProxy  # type: ignore[return-value]
