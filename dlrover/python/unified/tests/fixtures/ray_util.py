from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import ray

from dlrover.python.unified.util.actor_helper import kill_actors
from dlrover.python.unified.util.test_hooks import coverage_enabled


@pytest.fixture(scope="session", autouse=True)
def disable_ray_auto_init():
    def auto_init_ray():
        """Patch to disable ray auto init in tests."""
        assert ray.is_initialized(), (
            "Ray should be initialized before using Ray APIs."
        )

    with patch("ray._private.auto_init_hook.auto_init_ray", auto_init_ray):
        yield


@pytest.fixture(scope="session")
def coverage_envs():
    """Fixture to set up coverage environment variables."""
    envs = {}
    if coverage_enabled():
        file = Path(__file__).parent / "ray.coveragerc"
        envs["COVERAGE_PROCESS_START"] = file.as_posix()
        print("Coverage enabled, setting up environment variables for ray.")
        print(f"  COVERAGE_PROCESS_START={file.as_posix()}")
    yield envs
    if coverage_enabled():
        print("Combining coverage data...")
        import coverage

        coverage.Coverage.current().combine()


@contextmanager
def _setup_ray(envs):
    """Setup coverage environment variables if coverage is enabled."""
    if ray.is_initialized():
        pytest.fail("Ray is already initialized before setup.")
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


@pytest.fixture()
def tmp_ray(coverage_envs):
    """Fixture to initialize and shutdown Ray."""
    with _setup_ray(coverage_envs):
        yield


@pytest.fixture(scope="module")
def shared_ray(coverage_envs):
    """Fixture to initialize and shutdown Ray. Shared across tests.
    Module scope to avoid affecting other modules."""
    with _setup_ray(coverage_envs):
        yield
