#  Copyright 2025 The DLRover Authors. All rights reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import ray
from ray._private.test_utils import wait_for_condition

from dlrover.python.common.log import default_logger as logger


class BaseTest(unittest.TestCase):
    def setUp(self):
        logger.info(
            f"========= {self.__class__.__name__}-"
            f"{self._testMethodName} start ========="
        )

    def tearDown(self):
        logger.info(
            f"========= {self.__class__.__name__}-"
            f"{self._testMethodName} end ========="
        )


class AsyncBaseTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        logger.info(
            f"========= {self.__class__.__name__}-"
            f"{self._testMethodName} start ========="
        )

    def tearDown(self):
        logger.info(
            f"========= {self.__class__.__name__}-"
            f"{self._testMethodName} end ========="
        )


class RayBaseTest(BaseTest):
    @classmethod
    def init_ray_safely(cls, **kwargs):
        if ray.is_initialized():
            cls.close_ray_safely()
        ray.init(**kwargs)

    @classmethod
    def close_ray_safely(cls):
        ray.shutdown()
        wait_for_condition(lambda: not ray.is_initialized())
