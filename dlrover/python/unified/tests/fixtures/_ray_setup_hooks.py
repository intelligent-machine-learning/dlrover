import json
import os
from pathlib import Path

from dlrover.python.unified.util.test_hooks import coverage_enabled

_PARAM_ENV = "DLROVER_TEST_SETUP_HOOKS_ARGS"


def _setup_coverage():
    file = Path(__file__).parent / "_ray.coveragerc"
    try:
        import coverage

        os.environ["COVERAGE_PROCESS_START"] = file.as_posix()
        coverage.process_startup()
    except ImportError:
        pass


def get_args():
    module = str(get_args.__module__)
    params = {
        "setup_coverage": coverage_enabled(),
    }
    return f"{module}.hook", {_PARAM_ENV: json.dumps(params)}


def hook():
    """Main function to set up coverage."""
    params = json.loads(os.environ.get(_PARAM_ENV, "{}"))
    if params.get("setup_coverage", False):
        _setup_coverage()
