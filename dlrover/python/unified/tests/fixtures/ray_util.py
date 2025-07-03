from contextlib import contextmanager
from pathlib import Path

import pytest
import ray

from dlrover.python.unified.util.actor_helper import kill_actors
from dlrover.python.unified.util.test_hooks import coverage_enabled


@contextmanager
def _setup_ray():
    """Setup coverage environment variables if coverage is enabled."""
    envs = {}
    if coverage_enabled():
        file = Path(__file__).parent / "ray.coveragerc"
        envs["COVERAGE_PROCESS_START"] = file.as_posix()
        print("Coverage enabled, setting up environment variables for ray.")
        print(f"  COVERAGE_PROCESS_START={file.as_posix()}")

    ray.init(
        namespace="dlrover_test",
        runtime_env={"env_vars": envs},
    )
    yield

    actors = ray.util.list_named_actors()
    if actors:
        print(f"Cleaning up {len(actors)} actors...: {actors}")
        kill_actors(actors)
    ray.shutdown()
    if coverage_enabled():
        print("Combining coverage data...")
        import coverage

        coverage.Coverage().combine()


@pytest.fixture()
def tmp_ray():
    """Fixture to initialize and shutdown Ray."""
    with _setup_ray():
        yield


@pytest.fixture(scope="session")
def session_ray():
    """Fixture to initialize and shutdown Ray.
    Session-scoped, shared across tests."""
    with _setup_ray():
        yield
