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
from unittest.mock import patch

import pytest
import ray

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.tests.fixtures import _ray_setup_hooks
from dlrover.python.unified.util.actor_helper import (
    __actors_cache,
    kill_actors,
)
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


@pytest.fixture(scope="session", autouse=True)
def coverage_combine():
    yield
    if coverage_enabled():
        logger.info("Combining coverage data...")
        import coverage

        cur = coverage.Coverage.current()
        assert cur is not None
        cur.combine()


@contextmanager
def _setup_ray(**kwargs):
    """Setup coverage environment variables if coverage is enabled."""
    if ray.is_initialized():
        pytest.fail("Ray is already initialized before setup.")
    if len(kwargs) > 0:
        logger.warning(
            f"Ray is initialized with additional parameters: {kwargs}"
        )
    setup_hook, setup_envs = _ray_setup_hooks.get_args()
    print(f"Setting up ray setup hooks: {setup_hook}")
    ray.init(
        namespace="dlrover_test",
        num_cpus=8,  # Default CPU count for tests
        **kwargs,
        runtime_env={
            "env_vars": {**setup_envs},
            "worker_process_setup_hook": setup_hook,
            **kwargs.get("runtime_env", {}),
        },
    )
    yield

    actors = ray.util.list_named_actors()
    if actors:
        logger.warning(f"Cleaning up {len(actors)} actors...: {actors}")
        kill_actors(actors)
    __actors_cache.clear()
    ray.shutdown()


@pytest.fixture()
def tmp_ray(request):
    """Fixture to initialize and shutdown Ray."""
    options = request.param if hasattr(request, "param") else {}
    with _setup_ray(**options):
        yield


@pytest.fixture(scope="module")
def shared_ray():
    """Fixture to initialize and shutdown Ray. Shared across tests.
    Module scope to avoid affecting other modules.
    """
    with _setup_ray():
        yield
