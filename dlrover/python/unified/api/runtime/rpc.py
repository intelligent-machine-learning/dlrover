from typing import Any, Callable, Dict, Optional, ParamSpec

from gguf import TypeVar

from dlrover.python.unified.util.actor_helper import ActorInvocation
from dlrover.python.unified.util.actor_proxy import invoke_actor_t, invoke_meta

RPC_REGISTRY: Dict[str, Callable[..., Any]] = {}


def rpc(name: Optional[str] = None):
    """Decorator to mark a method as an RPC method."""
    # TODO More metadata

    def decorator(func):
        func._rpc_name = name or func.__name__
        RPC_REGISTRY[func._rpc_name] = func
        return func

    return decorator


@invoke_meta("_arbitrary_remote_call")
def _arbitrary_call(func, *args, **kwargs) -> Any:
    raise NotImplementedError("stub")  # pragma: no cover


@invoke_meta("_user_rpc_call")
def _user_rpc_call(fn_name: str, *args, **kwargs) -> Any:
    raise NotImplementedError("stub")  # pragma: no cover


P = ParamSpec("P")
R = TypeVar("R", covariant=True)


def rpc_call_t(
    actor: str, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> ActorInvocation[R]:
    """Call a method on a remote actor."""
    return invoke_actor_t(
        _user_rpc_call, actor, method.__name__, *args, **kwargs
    )


def rpc_call_arbitrary(
    actor: str, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
) -> ActorInvocation[R]:
    """Call an arbitrary method on a remote actor."""
    return invoke_actor_t(_arbitrary_call, actor, method, *args, **kwargs)
