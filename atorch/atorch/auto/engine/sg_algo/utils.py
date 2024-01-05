import json
import pickle
from typing import List

import numpy as np
import pandas as pd

from atorch.auto.engine.strategy import StrategyStatus


def analyse_strategies(finished_strategies):
    iterations = len(finished_strategies)
    throughput_list = []
    for strategy_info in finished_strategies.values():
        dryrun_result = strategy_info.dryrun_result
        if dryrun_result:
            throughput_list.append(dryrun_result["throughput"])
        else:
            throughput_list.append(0.0)
    throughput_arr = np.array(throughput_list)
    if len(throughput_list) == 0:
        patience = 0
    else:
        patience = int(iterations - np.argmax(throughput_arr) - 1)
    return iterations, patience


def rec_to_easydl_strategy(rec, wrap_cls, included_opts=[]) -> list:
    """
    rec: pd.DataFrame
    wrap_cls: tuple
    included_opts: List[str]
    """
    wrap_cls_in = wrap_cls
    for item in included_opts:
        if isinstance(item, dict):
            wrap_cls_inclued = item.get("atorch_wrap_cls", ())
            if isinstance(wrap_cls_inclued, str):
                wrap_cls_in = tuple([wrap_cls_inclued])
            elif isinstance(wrap_cls_inclued, tuple) and len(wrap_cls_inclued) > 0:
                wrap_cls_in = wrap_cls_inclued

    strategy_list = []
    for _, row in rec.iterrows():
        load_strategy = []
        for col, val in row.items():
            try:
                if isinstance(json.loads(val), dict):
                    parallel_mode_config = []
                    for item in json.loads(val).items():
                        if item[1] > 1:
                            parallel_mode_config.append((item[0], item[1]))
                            if item[0] == "tensor":
                                opt_name = "tensor_parallel"
                                strategy = (opt_name, pickle.dumps(None), True)
                                load_strategy.append(strategy)
                        else:
                            assert item[0] != "data"
                    config = pickle.dumps((parallel_mode_config, None))
                    load_strategy.append(("parallel_mode", config, False))
            except ValueError:
                if isinstance(val, str):
                    if val != "NotChosen":
                        if val == "zero1":
                            strategy = (val, pickle.dumps(None), False)
                            load_strategy.append(strategy)
                        # elif val == "zero2_fairscale":
                        #     config = pickle.dumps({"not_use_fsdp": True})
                        #     strategy = ("zero2", config, False)
                        #     load_strategy.append(strategy)
                        elif val == "zero2_fsdp":
                            load_strategy.append(
                                (
                                    "zero2",
                                    pickle.dumps(
                                        {
                                            "not_use_fsdp": False,
                                            "sync_module_states": True,
                                            "forward_prefetch": True,
                                            "limit_all_gathers": True,
                                            "atorch_wrap_cls": wrap_cls_in,
                                        }
                                    ),
                                    False,
                                )
                            )
                        elif val == "fsdp":
                            load_strategy.append(
                                (
                                    "fsdp",
                                    pickle.dumps(
                                        {
                                            "forward_prefetch": True,
                                            "limit_all_gathers": True,
                                            "sync_module_states": True,
                                            "atorch_wrap_cls": wrap_cls_in,
                                        }
                                    ),
                                    False,
                                )
                            )
                        elif val == "checkpoint":
                            config = pickle.dumps(wrap_cls_in)
                            load_strategy.append(("checkpoint", config, False))
                        else:
                            load_strategy.append((val, pickle.dumps(None), False))

        if ("half" in included_opts) or ("half", "fp16") in included_opts:
            load_strategy.append(("half", pickle.dumps("fp16"), False))
        elif ("half", "bf16") in included_opts:
            load_strategy.append(("half", pickle.dumps("bf16"), False))
        load_strategy_new: List = []
        for item in load_strategy:
            if item[0] == "module_replace":
                load_strategy_new.insert(0, item)
            else:
                load_strategy_new.append(item)

        strategy_list.append(load_strategy_new)
    return strategy_list


def get_finished_strategies(strategies: dict) -> dict:
    finished_strategies = {}
    success = StrategyStatus.SUCCEED
    fail = StrategyStatus.FAILED
    for s_id, strategy_info in strategies.items():
        if (strategy_info.status == success) or (strategy_info.status == fail):
            finished_strategies[s_id] = strategy_info
    return finished_strategies


def gen_space_config(opt_lib_group, opt_lib_methods, total_process, included_opts=[]):
    space_config = []
    # every group is a search variable in Bayesian Optimization

    # included_opts support following
    # 'amp_native'
    # 'zero1', 'zero2', 'fsdp', 'checkpoint' ,
    # 'half', ('half','bf16'), ('half','fp16')
    for group_name, opt_candidates_raw in opt_lib_group.items():
        opt_candidates = []

        # basic_prune tune can disable module_replace.
        # if module_replace is disabled, do not consider module_replace
        if group_name == "module_replace" and opt_lib_methods["module_replace"].disabled:
            continue

        if group_name == "parallel":
            continue
        # do not search half
        if group_name == "half":
            continue
        # if half is in included, amp is not considered
        if group_name == "amp" and (
            "half" in included_opts or ("half", "bf16") in included_opts or ("half", "fp16") in included_opts
        ):
            continue

        # zero only works when world_size>1
        if total_process == 1 and group_name == "zero":
            continue
        # parallel_mode only works when world_size>1
        if total_process == 1 and group_name == "parallel_mode":
            continue

        if group_name == "dynamo":
            continue

        for opt_name in opt_candidates_raw:

            # if the opt is included in this group, we only search this
            if opt_name in included_opts:
                opt_candidates = [opt_name]
                break

            if not opt_lib_methods[opt_name].disabled:
                opt_candidates.append(opt_name)

        # if world_size>1, we ensure to use ddp

        if group_name == "parallel_mode":
            if opt_lib_methods["tensor_parallel"].disabled:
                # if cpu is used, enter this branch
                space_config.append(
                    {
                        "name": group_name,
                        "type": "cat",
                        "categories": [json.dumps({"data": total_process, "tensor": 1})],
                    }
                )
            else:
                space_config.append(
                    {
                        "name": group_name,
                        "type": "cat",
                        "categories": [json.dumps({"data": total_process, "tensor": 1})],
                    }
                )
        elif group_name == "zero":

            if len(opt_candidates) > 0:

                if len(opt_candidates) == 1 and opt_candidates[0] in included_opts:
                    # this group has included op
                    if opt_candidates == "zero2":
                        # zero2 has fsdp and fairscale
                        space_config.append(
                            {
                                "name": group_name,
                                "type": "cat",
                                "categories": [
                                    "zero2_fsdp",
                                    # "zero2_fairscale",
                                ],
                            }
                        )
                    else:
                        space_config.append(
                            {
                                "name": group_name,
                                "type": "cat",
                                "categories": opt_candidates,
                            }
                        )
                else:
                    if "zero2" in opt_candidates:
                        opt_candidates.remove("zero2")
                        if group_name in included_opts:
                            var_cats = [
                                "zero2_fsdp",
                                # "zero2_fairscale",
                            ] + opt_candidates
                        else:
                            var_cats = (
                                [
                                    "zero2_fsdp",
                                    # "zero2_fairscale"
                                ]
                                + opt_candidates
                                + ["NotChosen"]
                            )

                        space_config.append(
                            {
                                "name": group_name,
                                "type": "cat",
                                "categories": var_cats,
                            }
                        )
                    else:
                        if group_name in included_opts:
                            var_cats = opt_candidates
                        else:
                            var_cats = opt_candidates + ["NotChosen"]

                        space_config.append(
                            {
                                "name": group_name,
                                "type": "cat",
                                "categories": var_cats,
                            }
                        )

        else:
            if len(opt_candidates) > 0:
                if len(opt_candidates) == 1 and opt_candidates[0] in included_opts:
                    space_config.append(
                        {
                            "name": group_name,
                            "type": "cat",
                            "categories": opt_candidates,
                        }
                    )
                else:
                    if group_name in included_opts:
                        var_cats = opt_candidates
                    else:
                        var_cats = opt_candidates + ["NotChosen"]
                    space_config.append(
                        {
                            "name": group_name,
                            "type": "cat",
                            "categories": var_cats,
                        }
                    )

    return space_config


def unfeasible_filter(rec_df, forbidden_columns, forbidden_list, n_suggestions=1):
    assert n_suggestions == 1, "Default only support one rec!!!"

    try:
        valid_rec_df = rec_df[
            rec_df[forbidden_columns]
            .apply(func=lambda x: "+".join(x.tolist()), axis=1)
            .apply(func=lambda x: x not in forbidden_list)
        ]
    except KeyError:
        valid_rec_df = rec_df

    if len(valid_rec_df) == 0:
        return rec_df.sample(n=n_suggestions), False
    else:
        return valid_rec_df.sample(n=n_suggestions), True


def transform_finished_strategies_to_hebo(space, opt_lib_group, finished_strategies, included_opts=[]):
    opt_to_group_hash = {}
    for group_name, opt_candidates in opt_lib_group.items():
        for opt_name in opt_candidates:
            assert not (opt_name in opt_to_group_hash.keys()), "We should not have duplicate opts in different groups!!"
            opt_to_group_hash[opt_name] = group_name

    X = pd.DataFrame(columns=space.para_names, dtype=object)
    y = []
    for strategy_info in finished_strategies.values():
        X_dict = {}
        for opt in strategy_info.strategy:
            if opt[0] == "parallel_mode":
                config = {item[0]: item[1] for item in pickle.loads(opt[1])[0]}
                if "tensor" not in config.keys():
                    config["tensor"] = 1
                config = {key: config[key] for key in ["data", "tensor"]}
                # default config['data']>1
                X_dict[opt_to_group_hash[opt[0]]] = json.dumps(config)
            elif opt[0] in ["zero1", "zero2", "fsdp"]:

                if opt[0] == "zero1":
                    X_dict[opt_to_group_hash[opt[0]]] = opt[0]
                elif opt[0] == "zero2":
                    zero_config = pickle.loads(opt[1])
                    if zero_config["not_use_fsdp"]:
                        X_dict[opt_to_group_hash[opt[0]]] = "zero2_fairscale"
                    else:
                        X_dict[opt_to_group_hash[opt[0]]] = "zero2_fsdp"
                else:
                    assert opt[0] == "fsdp"
                    X_dict[opt_to_group_hash[opt[0]]] = "fsdp"
            else:
                X_dict[opt_to_group_hash[opt[0]]] = opt[0]

        for group_name in space.para_names:
            if not (group_name in X_dict.keys()):
                X_dict[group_name] = "NotChosen"

        X = pd.concat([X, pd.DataFrame(X_dict, index=[0], dtype=object)], ignore_index=True, axis=0)

        dryrun_result = strategy_info.dryrun_result
        if dryrun_result:
            y.append(strategy_info.dryrun_result["throughput"])
        else:
            y.append(0.0)

    return X, -1 * np.array(y).reshape(-1, 1)
