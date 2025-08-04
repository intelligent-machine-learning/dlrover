from concurrent.futures import Future
from typing import Callable, List, ParamSpec, Protocol, Tuple, TypeVar

from dlrover.python.unified.api.runtime.rpc import rpc_call_arbitrary
from dlrover.python.unified.backend.common.base_worker import BaseWorker
from dlrover.python.unified.common.workload_base import (
    ActorInfo,
    JobInfo,
)
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.actor_proxy import invoke_actors
from dlrover.python.unified.util.async_helper import as_future


class Worker(Protocol):
    @property
    def job_info(self) -> JobInfo:
        """Get job information."""
        ...  # pragma: no cover

    @property
    def actor_info(self) -> ActorInfo:
        """Get actor information."""
        ...  # pragma: no cover


def current_worker() -> Worker:
    return BaseWorker.CURRENT


P = ParamSpec("P")
R = TypeVar("R", covariant=True)
T = TypeVar("T", covariant=True)


class RoleActor:
    def __init__(self, info: ActorInfo):
        self.name = info.name
        self.rank = info.rank
        self.info = info

    def call(
        self, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[R]:
        """Invoke a method on the actor."""
        ref = rpc_call_arbitrary(self.name, method, *args, **kwargs)
        return as_future(ref.async_wait())


class RoleGroup(Tuple["RoleActor", ...]):
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
        tuple.__init__(
            self,
            *[RoleActor(actor) for actor in actor_infos],
        )

    def call(
        self, method: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Future[List[R]]:
        """Invoke a method on all actors in the role group."""
        ref = invoke_actors(
            rpc_call_arbitrary(actor.name, method, *args, **kwargs)
            for actor in self
        )

        async def get_results():
            results = await ref.async_wait()
            return results.results

        return as_future(get_results())
