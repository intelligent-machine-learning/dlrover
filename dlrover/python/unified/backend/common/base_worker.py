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
from threading import Thread
from typing import ClassVar

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.api.runtime.rpc import (
    RPC_REGISTRY,
    export_rpc_instance,
)
from dlrover.python.unified.common.workload_base import ActorBase, WorkerStage
from dlrover.python.util.reflect_util import import_callable


class BaseWorker(ActorBase):
    """Base class for Worker actors in the DLRover system."""

    CURRENT: ClassVar["BaseWorker"]

    def _setup(self):
        BaseWorker.CURRENT = self

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
        try:
            logger.info(f"Entry point: {self.actor_info.spec.entry_point}")
            user_func = import_callable(self.actor_info.spec.entry_point)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import user function {self.actor_info.spec.entry_point}: {e}"
            ) from e

        # Export all module-level RPC methods.
        export_rpc_instance(None, user_func.__module__)
        if isinstance(user_func, type):
            logger.info(
                f"Instantiating user class {user_func} for actor {self.actor_info.name}."
            )
            try:
                inst = user_func()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate user class {user_func}"
                ) from e
            export_rpc_instance(None, inst)
            if not hasattr(inst, "run"):
                logger.error(
                    f"User class {user_func} does not have a 'run' method, "
                    "assert FINISH. This may cause job state inconsistency."
                )
                self._update_stage_force(
                    WorkerStage.FINISHED, WorkerStage.READY
                )
                return

            user_func = inst.run

        Thread(
            target=self._execute_user_function, args=(user_func,), daemon=True
        ).start()
        self._update_stage_force(WorkerStage.RUNNING, WorkerStage.READY)

    def _execute_user_function(self, user_func):
        """Execute the user function."""

        try:
            logger.info(f"Executing: {user_func}")
            inst = user_func()
            # If user function is a class, export all rpc methods.
            if isinstance(user_func, type) and inst is not None:
                export_rpc_instance(None, inst)
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
        while self.stage == WorkerStage.READY:
            logger.warning(
                f"Actor {self.actor_info.name} is starting, not ready for RPC {fn_name}, waiting..."
            )
            await asyncio.sleep(5)
        if self.stage not in [WorkerStage.RUNNING, WorkerStage.FINISHED]:
            raise RuntimeError(
                f"Actor {self.actor_info.name} is not in a valid state to handle RPC calls. "
                f"Current stage: {self.stage}. Expected RUNNING or FINISHED."
            )

        if fn_name not in RPC_REGISTRY:
            raise ValueError(
                f"RPC method {fn_name} not registered in {self.actor_info.name}."
            )
        func = RPC_REGISTRY[fn_name]
        return await self._arbitrary_remote_call(func, *args, **kwargs)

    async def _arbitrary_remote_call(self, fn, *args, **kwargs):
        """Handle an arbitrary remote call."""
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        else:
            return await asyncio.to_thread(fn, *args, **kwargs)
