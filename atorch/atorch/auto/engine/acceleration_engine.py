# Copyright 2022 The ElasticDL Authors. All rights reserved.
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
import time

from atorch.auto.engine.analyser_result import AnalyserResult
from atorch.auto.engine.executor import Executor
from atorch.auto.engine.optimization_method import OptimizationMethodLibrary
from atorch.auto.engine.planner import Planner
from atorch.auto.engine.servicer import create_acceleration_service
from atorch.auto.engine.sg_algo.sg_algo_lib import StrategyGenerationAlgorithmLibrary
from atorch.auto.engine.strategy import StrategyInfoCollection
from atorch.common.log_utils import default_logger as logger


class AccelerationEngine(object):
    def __init__(
        self,
        device_context,
        included_opts=None,
        excluded_opts=None,
        time_limit=None,
        load_strategy=None,
        verbose=False,
    ):
        self.executor = self.create_executor(
            device_context=device_context,
            included_opts=included_opts,
            excluded_opts=excluded_opts,
            time_limit=time_limit,
            load_strategy=load_strategy,
            verbose=verbose,
        )
        self.port = None

    @staticmethod
    def create_executor(
        device_context=None,
        included_opts=None,
        excluded_opts=None,
        time_limit=None,
        load_strategy=None,
        verbose=False,
    ):
        # init OptimizationMethodLibrary
        opt_method_lib = OptimizationMethodLibrary()
        # init StrategyGenerationAlgorithmLibrary
        algo_lib = StrategyGenerationAlgorithmLibrary()
        # init StrategyInfoCollection with OptimizationMethodLibrary
        strategy_infos = StrategyInfoCollection(opt_method_lib)
        # init AnalyzerResult
        analyser_result = AnalyserResult()
        # init Planner
        planner = Planner(
            opt_method_lib,
            algo_lib,
            strategy_infos,
            analyser_result,
            device_context,
            load_strategy,
            included_opts,
            excluded_opts,
        )
        # create Executor
        executor = Executor(
            opt_method_lib,
            algo_lib,
            strategy_infos,
            analyser_result,
            planner,
            device_context=device_context,
            included_opts=included_opts,
            time_limit=time_limit,
            verbose=verbose,
        )
        return executor

    def start_service(self, port):
        self.port = port
        self.service = create_acceleration_service(self.port, self.executor)
        self.service.start()
        logger.info(f"Auto-Acceleration Service has started at port {self.port}")

    def tear_down(self, force_stopping=False, timeout=120):
        logger.info("Stopping Auto-Acceleration Service")
        duration = 0
        while not self.executor.can_be_terminated and not force_stopping and duration < timeout:
            time.sleep(2)
            duration += 2
            if duration % 10 == 0:
                logger.info("Waiting all processes to get terminate task")
        if not self.executor.can_be_terminated:
            logger.warning("Not all processes get a terminate task!")
        self.service.stop(None)
        logger.info("Auto-Acceleration Service has stopped.")

    def service_port(self):
        return self.port
