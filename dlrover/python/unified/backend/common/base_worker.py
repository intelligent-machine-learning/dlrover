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
import runpy
import sys
import shlex

from threading import Thread
from typing import ClassVar, Optional, Callable, Any

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.api.runtime.rpc_helper import (
    RPC_REGISTRY,
    export_rpc_instance,
)
from dlrover.python.unified.backend.common.events import BaseWorkerEvents
from dlrover.python.unified.common.actor_base import ActorBase
from dlrover.python.unified.common.enums import (
    ExecutionResult,
    WorkloadEntrypointType,
)
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.decorators import log_execution
from dlrover.python.util.reflect_util import import_callable


class BaseWorker(ActorBase):
    """Base class for Worker actors in the DLRover system."""

    CURRENT: ClassVar["BaseWorker"]

    def setup(self):
        BaseWorker.CURRENT = self

        self._user_rpc_ready = asyncio.Event()
        self._setup_envs()
        self._self_check()

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
        entrypoint = self.actor_info.spec.entry_point
        entrypoint_type = self.actor_info.spec.entry_point_type
        logger.info(
            f"[{self.actor_info.name}] starting with entrypoint({entrypoint_type.name}): {entrypoint}."
        )

        if entrypoint_type == WorkloadEntrypointType.MODULE_FUNC:
            self._start_with_module_func(entrypoint)
        elif entrypoint_type == WorkloadEntrypointType.PY_CMD:
            self._start_with_py_cmd(entrypoint)

    def _start_with_module_func(self, entrypoint: str):
        try:
            with BaseWorkerEvents.import_user_entrypoint(entrypoint):
                user_func: Optional[Callable[..., Any]] = import_callable(
                    entrypoint
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to import user function {self.actor_info.spec.entry_point}: {e}"
            ) from e

        # Export all module-level RPC methods.
        import inspect

        export_rpc_instance(None, inspect.getmodule(user_func))
        if isinstance(user_func, type):
            logger.info(
                f"Instantiating user class {user_func} for actor {self.actor_info.name}."
            )
            try:
                with BaseWorkerEvents.instantiate_user_class(str(user_func)):
                    inst = user_func()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate user class {user_func}"
                ) from e
            export_rpc_instance(None, inst)
            user_func = getattr(inst, "run", None)
            if self.actor_info.spec.is_driver and user_func is None:
                raise ValueError(
                    f"User class {user_func} does not have a 'run' method."
                    "If this workload is RPC-driven, set `is_driver=False` in WorkLoad."
                )

        logger.info(f"Exported RPC methods: {list(RPC_REGISTRY.keys())}")
        self._user_rpc_ready.set()

        if user_func is not None:
            Thread(
                target=self._execute_user_function,
                args=(user_func,),
                daemon=True,
                name="user_main_thread",
            ).start()
        else:
            logger.warning("No user function to execute.")

    def _start_with_py_cmd(self, entrypoint):
        Thread(
            target=self._execute_user_command,
            args=(entrypoint,),
            daemon=True,
            name="user_main_thread",
        ).start()

    def _on_execution_end(self, result: ExecutionResult):
        """Report the execution result to the prime master."""
        PrimeMasterApi.report_execution_result(self.actor_info.name, result)

    def _execute_user_function(self, user_func):
        """Execute the user function."""

        try:
            with (
                log_execution(
                    f"user_function:{user_func}", log_exception=False
                ),
                BaseWorkerEvents.running(),
            ):
                user_func()
        except Exception:
            logger.exception(
                f"Unexpected error occurred while executing user function({user_func})."
            )
            self._on_execution_end(ExecutionResult.FAIL)
        else:
            self._on_execution_end(ExecutionResult.SUCCESS)

    def _execute_user_command(self, command: str):
        """Execute the user command."""

        try:
            parts = shlex.split(command)
            python_file = parts[0]
            args = parts[1:]

            # save original argv
            original_argv = sys.argv
            sys.argv = [python_file] + args

            try:
                with (
                    log_execution(
                        f"user_command:{command}", log_exception=False
                    ),
                    BaseWorkerEvents.running(),
                ):
                    # run py command by runpy
                    runpy.run_path(python_file, run_name="__main__")
            finally:
                sys.argv = original_argv
            self._on_execution_end(ExecutionResult.SUCCESS)
        except SystemExit as e:
            logger.info(f"User command exited with exit code: {e.code}")
            exit_code = 0 if e.code is None or e.code == 0 else 1
            self._on_execution_end(
                ExecutionResult.SUCCESS
                if exit_code == 0
                else ExecutionResult.FAIL
            )
        except Exception as e:
            logger.exception(f"Command execution failed: {e}")
            self._on_execution_end(ExecutionResult.FAIL)

    # rpc
    async def _user_rpc_call(self, fn_name: str, *args, **kwargs):
        """Call a user-defined RPC method."""
        if not self._user_rpc_ready.is_set():
            logger.warning(
                f"Actor {self.actor_info.name} is starting, not ready for RPC {fn_name}, waiting..."
            )
            with log_execution("wait_user_rpc_ready"):
                await self._user_rpc_ready.wait()

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
