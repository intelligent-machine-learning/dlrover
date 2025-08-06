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
)

from gguf import TypeVar

from dlrover.python.unified.common.workload_base import ActorInfo
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.actor_proxy import (
    invoke_actor_t,
    invoke_actors,
    invoke_meta,
)
from dlrover.python.unified.util.async_helper import as_future

RPC_REGISTRY: Dict[str, Callable[..., Any]] = {}


@dataclass
class UserRpcMethodMeta:
    """Metadata for RPC methods."""

    name: str
    is_async: bool

    ATTR_KEY: ClassVar[str] = "_rpc_meta_"


def rpc(name: Optional[str] = None, export: bool = False):
    """Decorator to mark a method as an RPC method.

    Args:
        name: The name of the RPC method, default func.__name__.
        export: Whether to export the method as an RPC method, default True if top-level function.
    """

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
    if not hasattr(func, UserRpcMethodMeta.ATTR_KEY):
        raise ValueError(
            f"Function {func.__name__} is not decorated with @rpc."
        )
    if name in RPC_REGISTRY:
        raise ValueError(f"RPC method {name} already registered.")
    RPC_REGISTRY[name] = func


@invoke_meta("_user_rpc_call")
def _user_rpc_call(fn_name: str, *args, **kwargs) -> Any:
    raise NotImplementedError("stub")  # pragma: no cover


P = ParamSpec("P")
R = TypeVar("R", covariant=True)
T = TypeVar("T", covariant=True)


def _rpc_call(actor: str, method: str, args, kwargs) -> Any:
    return invoke_actor_t(_user_rpc_call, actor, method, *args, **kwargs)


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

    def call(
        self, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]:
        """Invoke a method on the actor."""
        return as_future(
            _rpc_call(self.name, method.__name__, args, kwargs).async_wait()
        )


class RoleGroup(Sequence["RoleActor"]):
    """A group of actors with the same role."""

    def __init__(self, role: str, optional: bool = False):
        """Get the role group for a specific role."""
        try:
            actor_infos = PrimeMasterApi.get_workers_by_role(role)
        except ValueError:
            if not optional:
                raise
            actor_infos = []
        if not optional and len(actor_infos) == 0:
            raise ValueError(f"No actors found for role '{role}'.")
        self.actors = [RoleActor(actor) for actor in actor_infos]

    def __getitem__(self, index: int) -> "RoleActor":
        """Get an actor by index."""
        return self.actors[index]

    def __len__(self) -> int:
        """Get the number of actors in the role group."""
        return len(self.actors)

    def call(
        self, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[List[R]]:
        """Invoke a method on all actors in the role group."""
        name = method if isinstance(method, str) else method.__name__
        ref = invoke_actors(
            _rpc_call(actor.name, name, args, kwargs) for actor in self
        )

        async def get_results():
            results = await ref.async_wait()
            return results.results

        return as_future(get_results())

    def call_rank0(
        self, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]:
        """Invoke a method on the rank 0 actor in the role group."""
        if not self.actors:
            raise ValueError("No actors in the role group.")
        return self.actors[0].call(method, *args, **kwargs)

    def split_param(self, param: Sequence[T]) -> List[Sequence[T]]:
        """Split the parameter into sub-batches for each actor."""
        sub_batch_size = math.ceil(len(param) / len(self.actors))
        return [
            param[
                i * sub_batch_size : min((i + 1) * sub_batch_size, len(param))
            ]
            for i in range(len(self.actors))
        ]
