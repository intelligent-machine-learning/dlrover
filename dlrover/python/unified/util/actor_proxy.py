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


T = TypeVar("T", bound="type")


class ActorProxy:
    """A proxy class to interact with Ray actors as if they were local objects.

    Inherit from this class to create an actor proxy.
    - ACTOR_NAME could be defined in the subclass,
      which is used to bind the actor name to the class.
    - All public static methods are replaced with remote_call,
      binding them to the ACTOR_NAME class variable.
    - All class methods remain unchanged, for non-rpc methods.
    """

    ACTOR_NAME: ClassVar[str]

    @classmethod
    def _wrap(cls, method: Callable[..., Any]):
        actor_name = getattr(cls, "ACTOR_NAME", None)

        def remote_call(*args, **kwargs):
            if not actor_name:
                raise TypeError(
                    "ACTOR_NAME not defined, you must use bind(actor_name)."
                )
            ref = invoke_actor_t(method, actor_name, *args, **kwargs)
            return ref.wait()

        return remote_call

    @classmethod
    def _do_wrap(cls, method: Callable[..., Any]):
        method = getattr(method, "__origin__", method)
        wrapped = cls._wrap(method)
        wrapped = wraps(method)(wrapped)
        setattr(wrapped, "__origin__", method)
        return wrapped

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # Replace all public static methods with remote_call, binding them to ACTOR_NAME.
        cls2: Optional[type] = cls
        while cls2 is not None and cls2 != ActorProxy:
            for name, method in cls2.__dict__.items():
                if not callable(method) or name.startswith("__"):
                    continue
                if isinstance(method, classmethod):
                    continue  # allow classmethod to remain unchanged, for non-rpc methods
                if not isinstance(method, staticmethod):
                    raise TypeError(
                        f"Rpc Method {method} must be a static method. or classmethod for non-rpc methods."
                    )
                setattr(
                    cls,
                    name,
                    staticmethod(cls._do_wrap(method.__func__)),
                )
            cls2 = cls2.__base__

    @classmethod
    def bind(cls: T, actor_name: str) -> T:
        """Create a new bound proxy class with the given actor name."""

        class BoundActorProxy(cls):  # type: ignore[misc,valid-type]
            ACTOR_NAME = actor_name

        return BoundActorProxy  # type: ignore[return-value]
