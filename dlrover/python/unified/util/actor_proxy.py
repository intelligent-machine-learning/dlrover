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

from functools import wraps
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

P = ParamSpec("P")
R = TypeVar("R")
UNBOUND = "UNBOUND"


class ActorCall(Generic[P, R]):
    """New decorator style for actor call, support binding actor name to class.

    Based on descriptor protocol __get__, could __call__ directly.
    Support extra usage: bind, _call, async_call.
    """

    def __init__(self, func: Callable[P, R], actor: str = UNBOUND) -> None:
        self._func = getattr(func, "__func__", func)
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
        return (
            f"ActorCall(func={self._func.__qualname__}, actor={self._actor})"
        )

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
