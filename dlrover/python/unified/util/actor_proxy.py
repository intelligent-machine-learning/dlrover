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

from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Generic,
    ParamSpec,
    TypeVar,
    overload,
)

from dlrover.python.unified.util.actor_helper import (
    ActorInvocation,
    invoke_actor_t,
)

T = TypeVar("T", bound="type")


# TODO remove this, use actor_call instead.
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

    @classmethod
    def _replace_rpc_methods(cls, target, wrap_method) -> None:
        # Replace all public static methods with remote_call, binding them to ACTOR_NAME.
        for base in cls.__mro__:
            for name, method in base.__dict__.items():
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


P = ParamSpec("P")
R = TypeVar("R")
UNBOUND = "UNBOUND"


class ActorCall(Generic[P, R]):
    """New decorator style for actor call, support binding actor name to class.

    Based on descriptor protocol __get__, could __call__ directly.
    Support extra usage: bind, _call, async_call.
    """

    def __init__(self, func: Callable[P, R], actor: str = UNBOUND) -> None:
        self._func = func
        self._actor = actor
        wraps(func)(self)

    def bind(self, actor: str):
        return ActorCall[P, R](self._func, actor)

    def _call(self, *args: P.args, **kwds: P.kwargs) -> ActorInvocation[R]:
        """Internal method to create an ActorInvocation for the call."""
        if self._actor is UNBOUND:
            raise ValueError("ACTOR_NAME is not bound, call bind() first.")
        return invoke_actor_t(self._func, self._actor, *args, **kwds)

    def __repr__(self) -> str:
        return f"ActorCall(func={self._func}, actor={self._actor})"

    # Misc

    @property
    def __func__(self) -> Callable[P, R]:
        return self._func

    def __get__(self, instance, owner):
        actor = getattr(instance or owner, "ACTOR_NAME", None)
        assert isinstance(actor, str | None)
        return self.bind(actor or self._actor)

    def __call__(self, *args: P.args, **kwds: P.kwargs) -> R:
        return self._call(*args, **kwds).wait()

    async def async_call(self, *args: P.args, **kwds: P.kwargs) -> R:
        return await self._call(*args, **kwds)


@overload
def actor_call(func: Callable[P, R]) -> ActorCall[P, R]: ...
@overload
def actor_call(
    *, actor: str
) -> Callable[[Callable[P, R]], ActorCall[P, R]]: ...
def actor_call(func=None, *, actor: str = UNBOUND) -> Any:
    if func is None:
        return lambda f: ActorCall(f, actor)
    return ActorCall(func, actor)
