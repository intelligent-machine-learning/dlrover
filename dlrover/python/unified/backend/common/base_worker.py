# Copyright 2026 The DLRover Authors. All rights reserved.
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
from dlrover.python.unified.common.actor_base import (
    ActorBase,
    ExecutionResult,
    DiagnosticInfo,
)
from dlrover.python.unified.common.enums import (
    ExecutionResultType,
    WorkloadEntrypointType,
    DiagnosticInfoType,
)
from dlrover.python.unified.controller.api import PrimeMasterApi
from dlrover.python.unified.util.decorators import log_execution
from dlrover.python.util.reflect_util import import_callable

import ray
from ray.util.state import get_log


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

    def get_diagnostic(self) -> DiagnosticInfo:
        """
        Get the diagnostic info of the worker.
        
        This method collects actor logs and performs basic diagnosis based on
        log content to determine error type, code and reason.
        
        Returns:
            DiagnosticInfo: The diagnostic information including log content
            and error classification.
        """
        # Get actor_id for log retrieval
        actor_id = self.id
        log_content = ""
        
        # Try to fetch logs using ray.util.state.get_log
        if actor_id:
            try:
                log_lines = []
                for line in get_log(actor_id=actor_id):
                    log_lines.append(line)
                log_content = "".join(log_lines)
                # Limit log content size to avoid excessive memory usage
                max_log_size = 1024 * 1024  # 1MB
                if len(log_content) > max_log_size:
                    log_content = log_content[-max_log_size:]
                    log_content = "...[truncated]...\n" + log_content
            except Exception as e:
                logger.warning(f"Failed to get logs for actor {actor_id}: {e}")
                log_content = f"[Failed to retrieve logs: {e}]"
        else:
            log_content = "[Actor ID not available, cannot retrieve logs]"
        
        # Parse logs to determine error type, code and reason
        diag_type, code, reason = self._parse_diagnostic_info(log_content)
        
        return DiagnosticInfo(
            type=diag_type,
            code=code,
            reason=reason,
            log_content=log_content,
            # responsibility is determined by manager side based on timestamp
        )
    
    def _parse_diagnostic_info(self, log_content: str) -> tuple:
        """
        Parse log content to determine diagnostic type, error code and reason.
        
        Args:
            log_content: The actor log content.
            
        Returns:
            tuple: (DiagnosticInfoType, code, reason)
        """
        # Default: unknown error
        diag_type = DiagnosticInfoType.ERROR
        code = 1
        reason = "Unknown error occurred"
        
        if not log_content:
            return diag_type, code, reason
        
        log_lower = log_content.lower()
        
        # Check for CUDA OOM
        if any(pattern in log_lower for pattern in [
            "cuda out of memory", "cuda oom", "out of cuda memory",
            "runtimeerror: cuda", "torch.cuda.outofmemoryerror"
        ]):
            return DiagnosticInfoType.FATAL, 1001, "GPU memory insufficient"
        
        # Check for NCCL/Distributed communication errors
        if any(pattern in log_lower for pattern in [
            "nccl", "distributed", "connection reset", "connection refused",
            "broken pipe", "socket timeout", "rendezvous", "processgroup"
        ]):
            return DiagnosticInfoType.ERROR, 1002, "Distributed communication failure"
        
        # Check for system OOM killed
        if any(pattern in log_lower for pattern in [
            "killed", "oom", "out of memory", "signal 9", "sigkill"
        ]):
            return DiagnosticInfoType.FATAL, 3001, "Process killed by system OOM"
        
        # Check for user code exceptions (common patterns)
        if any(pattern in log_lower for pattern in [
            "exception", "error", "traceback", "failed"
        ]):
            return DiagnosticInfoType.ERROR, 2001, "User function execution failed"
        
        return diag_type, code, reason

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

    def _on_execution_end(self, result_type: ExecutionResultType):
        """Report the execution result to the prime master."""

        if result_type == ExecutionResultType.FAIL:
            PrimeMasterApi.report_diagnostic_info(
                self.actor_info.name, self.get_diagnostic()
            )

        PrimeMasterApi.report_execution_result(
            self.actor_info.name, ExecutionResult(result=result_type)
        )

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
            self._on_execution_end(ExecutionResultType.FAIL)
        else:
            self._on_execution_end(ExecutionResultType.SUCCESS)

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
            self._on_execution_end(ExecutionResultType.SUCCESS)
        except SystemExit as e:
            logger.info(f"User command exited with exit code: {e.code}")
            exit_code = 0 if e.code is None or e.code == 0 else 1
            self._on_execution_end(
                ExecutionResultType.SUCCESS
                if exit_code == 0
                else ExecutionResultType.FAIL
            )
        except Exception as e:
            logger.exception(f"Command execution failed: {e}")
            self._on_execution_end(ExecutionResultType.FAIL)

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
