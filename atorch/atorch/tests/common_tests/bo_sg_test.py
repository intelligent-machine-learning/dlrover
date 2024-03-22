import collections
import json
import pickle
import unittest

import pandas as pd
import torch

from atorch.auto.engine.optimization_method import OptimizationMethodLibrary
from atorch.auto.engine.sg_algo.hebo.design_space.design_space import DesignSpace
from atorch.auto.engine.sg_algo.utils import (
    analyse_strategies,
    gen_space_config,
    get_finished_strategies,
    rec_to_easydl_strategy,
    transform_finished_strategies_to_hebo,
    unfeasible_filter,
)
from atorch.auto.engine.strategy import StrategyStatus

StrategyInfo = collections.namedtuple(
    "StrategyInfo",
    "strategy status dryrun_result process_mode",
)


opt_method_lib = OptimizationMethodLibrary()


finished_strategies = {}
fsdp_config = {}
fsdp_config["sync_module_states"] = True
para_config = pickle.dumps(([("data", 2), ("tensor", 4)], None))
finished_strategies[0] = StrategyInfo(
    [
        ("parallel_mode", pickle.dumps(([("data", 8)], None)), False),
        (
            "fsdp",
            pickle.dumps(fsdp_config),
            False,
        ),
    ],
    StrategyStatus.SUCCEED,
    {"throughput": 30.0},
    "ONE_PROCESS",
)
finished_strategies[1] = StrategyInfo(
    [
        ("zero1", pickle.dumps(None), False),
        ("parallel_mode", para_config, False),
    ],
    StrategyStatus.SUCCEED,
    {"throughput": 135.0},
    "ONE_PROCESS",
)

finished_strategies[2] = StrategyInfo(
    [
        ("zero1", pickle.dumps(None), False),
        ("parallel_mode", para_config, False),
    ],
    StrategyStatus.SUCCEED,
    {"throughput": 65.0},
    "ONE_PROCESS",
)

finished_strategies[3] = StrategyInfo(
    [
        ("parallel_mode", para_config, False),
        (
            "fsdp",
            pickle.dumps(fsdp_config),
            False,
        ),
    ],
    StrategyStatus.SUCCEED,
    {"throughput": 33.0},
    "ONE_PROCESS",
)
finished_strategies[4] = StrategyInfo(
    [
        ("amp_native", pickle.dumps(None), False),
        ("parallel_mode", para_config, False),
    ],
    StrategyStatus.SUCCEED,
    {"throughput": 200.0},
    "ONE_PROCESS",
)
finished_strategies[5] = StrategyInfo(
    [
        ("parallel_mode", para_config, False),
    ],
    StrategyStatus.SUCCEED,
    {"throughput": 11.0},
    "ONE_PROCESS",
)
finished_strategies[6] = StrategyInfo(
    [
        ("parallel_mode", para_config, False),
    ],
    StrategyStatus.FAILED,
    None,
    "ONE_PROCESS",
)
finished_strategies[7] = StrategyInfo(
    [("parallel_mode", pickle.dumps(([("data", 8)], None)), False)],
    StrategyStatus.FAILED,
    None,
    "ONE_PROCESS",
)


class bosgTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_analyse_strategies(self):
        iterations, patience = analyse_strategies(finished_strategies)
        self.assertEqual(iterations, 8)
        self.assertEqual(patience, 3)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_get_finished_strategies(self):
        got_finished_strategies = get_finished_strategies(finished_strategies)
        self.assertEqual(got_finished_strategies, finished_strategies)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_transform_rec_to_easydl_strategy(self):
        parallel_mode = json.dumps({"data": 4, "tensor": 2})
        rec_df = pd.DataFrame(
            data=[
                ["amp_native", "fsdp", parallel_mode, "NotChosen"],
                ["amp_native", "NotChosen", parallel_mode, "module_replace"],
            ],
            columns=["amp", "zero", "parallel_mode", "module_replace"],
        )
        strategy_list = rec_to_easydl_strategy(rec_df, wrap_cls=("GPT2Block",))

        self.assertEqual(len(strategy_list), 2)
        for strategy in strategy_list:
            self.assertEqual(isinstance(strategy, list), True)
            for opt in strategy:
                self.assertEqual(isinstance(opt, tuple), True)
                self.assertEqual(len(opt), 3)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_gen_space_config(self):

        space_config = gen_space_config(opt_method_lib.groups, opt_method_lib.methods, 1)
        names = [item["name"] for item in space_config]
        self.assertEqual("zero" not in names, True)
        self.assertEqual("parallel_mode" not in names, True)

        space_config = gen_space_config(opt_method_lib.groups, opt_method_lib.methods, 4)
        names = [item["name"] for item in space_config]
        self.assertEqual("zero" in names, True)
        self.assertEqual("parallel_mode" in names, True)

        space_config = gen_space_config(
            opt_method_lib.groups,
            opt_method_lib.methods,
            4,
            ["checkpoint"],
        )
        for item in space_config:
            if item["name"] == "checkpoint":
                self.assertEqual("NotChosen" not in item["categories"], True)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_transform_finished_strategies_to_hebo(self):
        space_config = gen_space_config(opt_method_lib.groups, opt_method_lib.methods, 4)

        space = DesignSpace().parse(space_config)

        X, y = transform_finished_strategies_to_hebo(space, opt_method_lib.groups, finished_strategies)

        self.assertEqual(len(X.columns.tolist()), 5)
        self.assertEqual(len(X), 8)
        self.assertEqual(len(y), 8)

    @unittest.skipIf(torch.cuda.is_available(), "run on cpu only")
    def test_unfeasible_filter(self):
        NOT_COMPATIBLE_OPTS = []

        space_config = gen_space_config(opt_method_lib.groups, opt_method_lib.methods, 8)
        space = DesignSpace().parse(space_config)
        rec_df = space.sample(num_samples=20)
        rec, status = unfeasible_filter(rec_df, ["amp", "zero"], NOT_COMPATIBLE_OPTS)
        self.assertEqual(len(rec), 1)
        for _, row in rec[["amp", "zero"]].iterrows():
            vals = []
            for col, val in row.items():
                vals.append(val)
        self.assertEqual("+".join(vals) not in NOT_COMPATIBLE_OPTS, True)


if __name__ == "__main__":
    unittest.main()
