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
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Iterable,
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

# Used to replace `self` in type hints for non-static methods.
# Will be ignored when passed as an argument.
SELF: Any = object()


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
    # Remove `self` if it's the first argument.
    if args and args[0] is SELF:
        args = args[1:]  # type: ignore[assignment]
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
    # Remove `self` if it's the first argument.
    if args and args[0] is SELF:
        args = args[1:]  # type: ignore[assignment]
    return invoke_actors(
        ActorInvocation[R](
            actor if isinstance(actor, str) else actor.name,
            name,
            *args,
            **kwargs,
        )
        for actor in actors
    )


def invoke_actors(
    refs: Iterable[ActorInvocation[R]],
) -> ActorBatchInvocation[R]:
    """Create a batch invocation from multiple ActorInvocation instances."""
    return ActorBatchInvocation[R](list(refs))


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

    ACTOR_NAME: str

    def __init__(self, actor_name: str) -> None:
        self.ACTOR_NAME = actor_name
        self._replace_rpc_methods(
            self, lambda m: partial(self._invoke, actor_name, m)
        )

    @classmethod
    def _make_unbound_invoke(cls, method: Callable[..., Any]):
        def unbound_invoke(*args, **kwargs):
            raise TypeError(f"ACTOR_NAME not defined, can't invoke {method}.")

        return unbound_invoke

    @classmethod
    def _invoke(
        cls, actor_name: str, method: Callable[..., Any], *args, **kwargs
    ):
        """Invoke a method on the actor with the given name."""
        ref = invoke_actor_t(method, actor_name, *args, **kwargs)
        return ref.wait()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        actor_name = getattr(cls, "ACTOR_NAME", None)
        if actor_name is None:
            cls._replace_rpc_methods(cls, cls._make_unbound_invoke)
        else:
            cls._replace_rpc_methods(
                cls, lambda m: partial(cls._invoke, actor_name, m)
            )
        for name, method in cls.__dict__.items():
            print(f"ActorProxy: {name} -> {method}")

    @classmethod
    def _replace_rpc_methods(cls, target, wrap_method) -> None:
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

                method = method.__func__
                method = getattr(method, "__origin__", method)
                wrapped = wrap_method(method)
                wrapped = wraps(method)(wrapped)
                setattr(wrapped, "__origin__", method)

                setattr(target, name, staticmethod(wrapped))
            cls2 = cls2.__base__
