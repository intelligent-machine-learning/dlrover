import asyncio
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Optional,
    ParamSpec,
    Type,
)

from gguf import TypeVar

from dlrover.python.unified.util.actor_helper import ActorInvocation
from dlrover.python.unified.util.actor_proxy import invoke_actor_t, invoke_meta

RPC_REGISTRY: Dict[str, Callable[..., Any]] = {}


@dataclass
class UserRpcMethodMeta:
    """Metadata for RPC methods."""

    name: str
    isAsync: bool

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
            isAsync=asyncio.iscoroutinefunction(func),
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


def export_rpc_instance(ns: Optional[str], instance: Any):
    """Export an instance's RPC methods."""
    for attr_name in dir(instance):
        func = getattr(instance, attr_name)
        if callable(func) and hasattr(func, UserRpcMethodMeta.ATTR_KEY):
            meta: UserRpcMethodMeta = getattr(func, UserRpcMethodMeta.ATTR_KEY)
            name = f"{ns}.{meta.name}" if ns else meta.name
            export_rpc_method(name, func)


@invoke_meta("_arbitrary_remote_call")
def _arbitrary_call(func: Callable[..., Any], *args, **kwargs) -> Any:
    raise NotImplementedError("stub")  # pragma: no cover


@invoke_meta("_user_rpc_call")
def _user_rpc_call(fn_name: str, *args, **kwargs) -> Any:
    raise NotImplementedError("stub")  # pragma: no cover


P = ParamSpec("P")
R = TypeVar("R", covariant=True)


def _rpc_call(actor: str, method: str, is_async: bool, args, kwargs) -> Any:
    ref = invoke_actor_t(_user_rpc_call, actor, method, *args, **kwargs)
    if is_async:
        return ref.async_wait()
    else:
        return ref.wait()


def rpc_call_t(
    actor: str, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> R:
    """Call a method on a remote actor."""
    if not hasattr(method, UserRpcMethodMeta.ATTR_KEY):
        raise ValueError(
            f"Method {method.__name__} is not decorated with @rpc."
        )
    meta: UserRpcMethodMeta = getattr(method, UserRpcMethodMeta.ATTR_KEY)
    return _rpc_call(actor, meta.name, meta.isAsync, args, kwargs)


def rpc_call_arbitrary(
    actor: str, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> ActorInvocation[R]:
    """Call an arbitrary method on a remote actor."""
    return invoke_actor_t(_arbitrary_call, actor, method, *args, **kwargs)


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
        return _rpc_call(self.owner, name, meta.isAsync, args, kwargs)


def create_rpc_proxy(owner: str, name: str, cls: Type[R]) -> R:
    """Create a proxy for a remote actor's RPC methods."""
    return UserRpcProxy(owner, name, cls)  # type: ignore
