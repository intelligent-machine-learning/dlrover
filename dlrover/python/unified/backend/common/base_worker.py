import asyncio
from threading import Thread

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.api.runtime.rpc import RPC_REGISTRY
from dlrover.python.unified.common.workload_base import ActorBase, WorkerStage
from dlrover.python.util.reflect_util import import_callable


class BaseWorker(ActorBase):
    """Base class for Worker actors in the DLRover system."""

    def _setup(self):
        self._setup_envs()
        self._self_check()
        self._update_stage_force(WorkerStage.READY, WorkerStage.INIT)

    def _setup_envs(self):
        """Setup environment variables for the worker."""
        # This method can be overridden by subclasses to set up specific environment variables.
        pass

    def _self_check(self):
        """Check the worker itself."""
        # This method can be overridden by subclasses to perform self-checks.
        logger.info(f"[{self.actor_info.name}] Running self check.")

    def start(self):
        """Start the worker."""
        # This method can be overridden by subclasses to implement specific start logic.
        logger.info(f"[{self.actor_info.name}] Starting.")
        Thread(target=self._execute_user_function, daemon=True).start()
        self._update_stage_force(WorkerStage.RUNNING, WorkerStage.READY)

    def _execute_user_function(self):
        """Execute the user function."""

        try:
            logger.info(f"Entry point: {self.actor_info.spec.entry_point}")
            user_func = import_callable(self.actor_info.spec.entry_point)
            logger.info(f"Executing entry point: {user_func}")
            user_func()
        except Exception:
            logger.error(
                "Unexpected error occurred while executing user function.",
                exc_info=True,
            )
            self._update_stage_force(WorkerStage.FAILED, WorkerStage.RUNNING)
        else:
            self._update_stage_force(WorkerStage.FINISHED, WorkerStage.RUNNING)

    # rpc
    async def _user_rpc_call(self, fn_name: str, *args, **kwargs):
        """Call a user-defined RPC method."""
        if fn_name not in RPC_REGISTRY:
            raise ValueError(
                f"RPC method {fn_name} not registered in {self.actor_info.name}."
            )
        func = RPC_REGISTRY[fn_name]
        ret = func(*args, **kwargs)
        if asyncio.iscoroutine(ret):
            return await ret
        return ret

    async def _arbitrary_remote_call(self, fn, *args, **kwargs):
        """Handle an arbitrary remote call."""
        ret = fn(*args, **kwargs)
        if asyncio.iscoroutine(ret):
            return await ret
        return ret
