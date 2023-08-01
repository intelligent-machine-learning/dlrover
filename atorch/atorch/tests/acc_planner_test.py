import pickle
import unittest

from atorch.auto.engine.analyser_result import AnalyserResult
from atorch.auto.engine.optimization_method import OptimizationMethodLibrary
from atorch.auto.engine.planner import Planner, PlannerStage
from atorch.auto.engine.sg_algo.sg_algo_lib import StrategyGenerationAlgorithmLibrary
from atorch.auto.engine.strategy import StrategyInfoCollection


class TestPlanner(unittest.TestCase):
    def test_planner(self):
        device_context = {
            "node_num": 1,
            "nproc_per_node": 2,
            "gpu_compute_capability": "5.0",
        }
        opt_method_lib = OptimizationMethodLibrary()
        opt_method_lib.add_methods()
        algo_lib = StrategyGenerationAlgorithmLibrary()
        strategy_infos = StrategyInfoCollection(opt_method_lib)
        analyser_result = AnalyserResult()
        zero1_strategy = [("zero1", pickle.dumps(None), False)]
        status, _ = opt_method_lib.validate_strategy(zero1_strategy)
        self.assertTrue(status)
        excluded_opts = ["zero"]
        planner = Planner(
            opt_method_lib,
            algo_lib,
            strategy_infos,
            analyser_result,
            device_context,
            excluded_opts=excluded_opts,
        )
        status, _ = opt_method_lib.validate_strategy(zero1_strategy)
        self.assertFalse(status)
        amp_native_strategy = [("amp_native", pickle.dumps(None), False)]
        status, _ = opt_method_lib.validate_strategy(amp_native_strategy)
        self.assertFalse(status)
        is_done, tasks, new_strategy_num, _ = planner.plan()
        self.assertFalse(is_done)
        self.assertEqual(planner.stage, PlannerStage.ANALYSE)
        self.assertTrue(len(tasks) > 0)
        self.assertTrue(new_strategy_num == 0)

        is_done, tasks, new_strategy_num, _ = planner.plan()
        self.assertFalse(is_done)
        self.assertEqual(planner.stage, PlannerStage.BASELINE_STRATEGY)
        self.assertTrue(tasks is None)
        self.assertTrue(new_strategy_num == 1)
        strategy = strategy_infos.get_baseline_strategy()
        self.assertTrue(strategy is not None)

        is_done, tasks, new_strategy_num, algos = planner.plan()
        self.assertTrue(is_done)
        self.assertEqual(planner.stage, PlannerStage.SELECT_ALGO)
        self.assertTrue(tasks is None)
        self.assertTrue(new_strategy_num == 0)
        strategy = strategy_infos.get_baseline_strategy()
        self.assertTrue(algos is not None and len(algos) > 0)

        is_done, _, _, _ = planner.plan()
        self.assertTrue(is_done)

        strategy_infos = StrategyInfoCollection(opt_method_lib)
        planner = Planner(
            opt_method_lib,
            algo_lib,
            strategy_infos,
            analyser_result,
            device_context,
            load_strategy=zero1_strategy,
            excluded_opts=excluded_opts,
        )
        is_done, tasks, new_strategy_num, algos = planner.plan()
        self.assertTrue(is_done)
        self.assertEqual(algos, [])
        self.assertTrue(new_strategy_num == 1)
