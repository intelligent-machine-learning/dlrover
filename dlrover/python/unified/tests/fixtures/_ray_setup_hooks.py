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

import os
from pathlib import Path
from typing import Callable, Dict

from dlrover.python.unified.util.test_hooks import coverage_enabled
from dlrover.python.util.reflect_util import import_callable

_ENV_PREFIX = "DLROVER_TEST_HOOKS_"
_ENV_COVERAGE = _ENV_PREFIX + "COVERAGE"
_ENV_CUSTOM_HOOKS = _ENV_PREFIX + "CUSTOM_HOOKS"


def _setup_coverage():
    file = Path(__file__).parent / "_ray.coveragerc"
    try:
        import coverage

        os.environ["COVERAGE_PROCESS_START"] = file.as_posix()
        coverage.process_startup()
    except ImportError:
        pass


def _setup_custom_hooks():
    """Set up custom hooks for testing."""
    custom_hooks = os.environ.get(_ENV_CUSTOM_HOOKS, "")
    if custom_hooks:
        for hook in custom_hooks.split(","):
            if not hook.strip():
                continue
            print(f"Executing custom hook: {hook}")
            try:
                hook = import_callable(hook)
                hook()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to execute custom hook: {hook}"
                ) from e


def get_args():
    module = str(get_args.__module__)
    return f"{module}.hook", {
        _ENV_COVERAGE: ("1" if coverage_enabled() else "0")
    }


def inject_hook(*hooks: str | Callable) -> Dict[str, str]:
    """Inject a custom hook callable into the environment."""

    def to_str(hook: Callable | str) -> str:
        if isinstance(hook, str):
            return hook
        """Convert a callable to its string representation."""
        if "." in hook.__qualname__:
            raise ValueError("Hook should be a module-level function.")
        return f"{hook.__module__}.{hook.__name__}"

    hook_ids = map(to_str, hooks)
    return {_ENV_CUSTOM_HOOKS: ",".join(hook_ids)}


def hook():
    """Main function to set up coverage."""
    if os.environ.get(_ENV_PREFIX + "COVERAGE", "0") == "1":
        # If coverage is enabled, set up the coverage environment.
        _setup_coverage()
    _setup_custom_hooks()
