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

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
import ray

from dlrover.python.common.log import default_logger as logger
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
        logger.info(
            "Coverage enabled, setting up environment variables for ray."
        )
        logger.info(f"  COVERAGE_PROCESS_START={file.as_posix()}")
    yield envs
    if coverage_enabled():
        logger.info("Combining coverage data...")
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
        logger.warning(f"Cleaning up {len(actors)} actors...: {actors}")
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
