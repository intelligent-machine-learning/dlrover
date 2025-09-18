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
import math
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    ParamSpec,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

from dlrover.python.unified.common.actor_base import ActorInfo
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.actor_helper import (
    invoke_actor,
    invoke_meta,
    wait_batch_invoke,
)
from dlrover.python.unified.util.async_helper import (
    as_future,
    completed_future,
)
from dlrover.python.unified.util.test_hooks import after_test_cleanup

RPC_REGISTRY: Dict[str, Callable[..., Any]] = {}
after_test_cleanup(RPC_REGISTRY.clear)


@dataclass
class UserRpcMethodMeta:
    """Metadata for RPC methods."""

    name: str
    is_async: bool

    ATTR_KEY: ClassVar[str] = "_rpc_meta_"


def rpc(name: Optional[Union[str, Callable]] = None, export: bool = False):
    """Decorator to mark a method as an RPC method.

    Args:
        name: The name of the RPC method, default func.__name__.
        export: Whether to export the method as an RPC method, default True if top-level function.
    """

    # use stub function name if provided
    if callable(name):
        name = name.__name__

    def decorator(func):
        meta = UserRpcMethodMeta(
            name=name or func.__name__,
            is_async=asyncio.iscoroutinefunction(func),
        )
        setattr(func, UserRpcMethodMeta.ATTR_KEY, meta)
        if export:
            export_rpc_method(meta.name, func)
        return func

    return decorator


def export_rpc_method(name: str, func: Callable[..., Any]):
    """Export a method as an RPC method."""
    if name in RPC_REGISTRY:
        raise ValueError(f"RPC method '{name}' already registered.")
    RPC_REGISTRY[name] = func


@invoke_meta("_user_rpc_call")
def _user_rpc_call(fn_name: str, *args, **kwargs) -> Any:
    raise NotImplementedError("stub")  # pragma: no cover


P = ParamSpec("P")
R = TypeVar("R", covariant=True)
T = TypeVar("T", covariant=True)


def _rpc_call(actor: str, method: str, args, kwargs):
    return invoke_actor(
        _user_rpc_call,
        actor,
        method,
        *args,
        **kwargs,
        _rpc_display_name=f"user_rpc({method})",
    )


# region UserRpcProxy


def export_rpc_instance(ns: Optional[str], instance: Any):
    """Export an instance's RPC methods."""
    for attr_name in dir(instance):
        func = getattr(instance, attr_name)
        if callable(func) and hasattr(func, UserRpcMethodMeta.ATTR_KEY):
            meta: UserRpcMethodMeta = getattr(func, UserRpcMethodMeta.ATTR_KEY)
            name = f"{ns}.{meta.name}" if ns else meta.name
            export_rpc_method(name, func)


class UserRpcProxy:
    def __init__(self, owner: str, name: str, cls: Type[R]) -> None:
        self.owner = owner
        self.name = name
        self.cls = cls
        for attr_name in dir(cls):
            if callable(getattr(cls, attr_name)) and hasattr(
                getattr(cls, attr_name), UserRpcMethodMeta.ATTR_KEY
            ):
                meta: UserRpcMethodMeta = getattr(
                    getattr(cls, attr_name), UserRpcMethodMeta.ATTR_KEY
                )
                setattr(self, attr_name, partial(self._rpc_call, meta))

    def _rpc_call(self, meta: UserRpcMethodMeta, *args, **kwargs):
        """Call a method on the remote actor."""
        name = f"{self.name}.{meta.name}"
        ref = _rpc_call(self.owner, name, args, kwargs)
        if meta.is_async:
            return ref.async_wait()
        else:
            return ref.wait()


def create_rpc_proxy(owner: str, name: str, cls: Type[R]) -> R:
    """Create a proxy for a remote actor's RPC methods."""
    return UserRpcProxy(owner, name, cls)  # type: ignore


# region RoleGroup


class RoleActor:
    def __init__(self, info: ActorInfo):
        self.name = info.name
        self.rank = info.rank
        self.info = info

    @overload
    def call(
        self, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]: ...
    @overload
    def call(self, method: str, *args: Any, **kwargs: Any) -> Future[Any]: ...
    def call(self, method, *args, **kwargs) -> Future[Any]:
        """Invoke a method on the actor."""
        name = method if isinstance(method, str) else method.__name__
        return as_future(_rpc_call(self.name, name, args, kwargs).async_wait())


class RoleGroup(Sequence["RoleActor"]):
    """A group of actors with the same role."""

    def __init__(self, role: str, optional: bool = False):
        """Get the role group for a specific role."""
        try:
            actor_infos = PrimeMasterApi.get_workers_by_role(
                role, optional=optional
            )
        except ValueError:
            if not optional:
                raise
            actor_infos = []
        if not optional and len(actor_infos) == 0:
            raise ValueError(f"No actors found for role '{role}'.")
        self.actors = [RoleActor(actor) for actor in actor_infos]

    @overload
    def __getitem__(self, index: int) -> "RoleActor": ...  # pragma: no cover
    @overload
    def __getitem__(
        self, index: slice
    ) -> Sequence["RoleActor"]: ...  # pragma: no cover
    def __getitem__(self, index):
        """Get an actor by index."""
        return self.actors[index]

    def __len__(self) -> int:
        """Get the number of actors in the role group."""
        return len(self.actors)

    @overload
    def call(
        self, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[List[R]]: ...  # pragma: no cover
    @overload
    def call(
        self, method: str, *args: Any, _scatter: bool = False, **kwargs: Any
    ) -> Future[List[Any]]: ...  # pragma: no cover
    def call(
        self, method, *args, _scatter=False, **kwargs
    ) -> Future[List[Any]]:
        """Invoke a method on all actors in the role group."""
        name = method if isinstance(method, str) else method.__name__

        if len(self.actors) == 0:
            return completed_future([])

        if _scatter:
            length = len(self.actors)
            assert (
                all(isinstance(arg, list) for arg in args)
                and all(isinstance(kwarg, list) for kwarg in kwargs.values())
                and all(len(arg) == length for arg in args)
                and all(len(kwarg) == length for kwarg in kwargs.values())
            ), "All arguments must be lists of the same length."
            calls = []
            for i in range(length):
                sliced_args = tuple(arg[i] for arg in args)
                sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                calls.append(
                    _rpc_call(
                        self.actors[i].name, name, sliced_args, sliced_kwargs
                    )
                )
        else:
            calls = [
                _rpc_call(actor.name, name, args, kwargs) for actor in self
            ]

        async def get_results():
            res = await wait_batch_invoke(calls)
            return res.results

        return as_future(get_results())

    @overload
    def call_rank0(
        self, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]: ...  # pragma: no cover
    @overload
    def call_rank0(
        self, method: str, *args: Any, **kwargs: Any
    ) -> Future[object]: ...  # pragma: no cover
    def call_rank0(self, method, *args, **kwargs) -> Future[Any]:
        """Invoke a method on the rank 0 actor in the role group."""
        if not self.actors:
            raise ValueError("No actors in the role group.")
        return self.actors[0].call(method, *args, **kwargs)

    def call_batch(
        self,
        method: Callable[P, Sequence[R]],
        size: int,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> "FutureSequence[R]":
        """Invoke a method on all actors in the role group with batched arguments."""
        if not self.actors:
            raise ValueError("No actors in the role group.")
        assert all(
            len(arg) == size for arg in args if isinstance(arg, Sequence)
        ) and all(len(v) == size for v in kwargs if isinstance(v, Sequence)), (
            "All Sequence arguments must have the same length as size."
        )
        sub_batch_size = math.ceil(size / len(self.actors))
        futures = []
        lens = []
        for i in range(len(self.actors)):
            start = i * sub_batch_size
            end = min((i + 1) * sub_batch_size, size)
            lens.append(end - start)
            sub_args = [
                arg[start:end] if isinstance(arg, Sequence) else arg
                for arg in args
            ]
            sub_kwargs = {
                k: v[start:end] if isinstance(v, Sequence) else v
                for k, v in kwargs.items()
            }
            futures.append(
                self.actors[i].call(method, *sub_args, **sub_kwargs)  # type: ignore
            )
        return FutureSequence(futures, lens)


class FutureSequence(Sequence[T]):
    """A wrapper for futures, providing a sequence-like interface."""

    def __init__(self, futures: List[Future[Sequence[T]]], lens: List[int]):
        self._lens = lens
        self._futures = futures

    @overload
    def __getitem__(self, index: int) -> T: ...  # pragma: no cover
    @overload
    def __getitem__(self, index: slice) -> Sequence[T]: ...  # pragma: no cover
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        future_index = 0
        while index >= self._lens[future_index]:
            index -= self._lens[future_index]
            future_index += 1
        return self._futures[future_index].result()[index]

    def __len__(self) -> int:
        return sum(self._lens)

    def __repr__(self) -> str:
        return f"FutureSequence(lens={self._lens})"

    def wait(self) -> List[T]:
        """Wait for all futures to complete and return the results."""
        results: List[T] = []
        for future in self._futures:
            results.extend(future.result())
        return results
