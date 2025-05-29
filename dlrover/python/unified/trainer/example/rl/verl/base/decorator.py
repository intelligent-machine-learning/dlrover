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
#
# This package includes code from [https://github.com/volcengine/verl]
# licensed under the Apache License 2.0. See [https://github.com/volcengine/
# verl] for details.

import inspect
from enum import Enum
from functools import wraps
from types import FunctionType
from typing import Dict, List, Tuple

from verl.protocol import DataProtoFuture, _padding_size_key

# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = "attrs_3141562937"


class Dispatch(Enum):
    RANK_ZERO = 0
    ONE_TO_ALL = 1
    ALL_TO_ALL = 2
    MEGATRON_COMPUTE = 3
    MEGATRON_PP_AS_DP = 4
    MEGATRON_PP_ONLY = 5
    MEGATRON_COMPUTE_PROTO = 6
    MEGATRON_PP_AS_DP_PROTO = 7
    DP_COMPUTE = 8
    DP_COMPUTE_PROTO = 9
    DP_COMPUTE_PROTO_WITH_FUNC = 10
    DP_COMPUTE_METRIC = 11

    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    DIRECT_ROLLOUT_METHOD = 12


class Execute(Enum):
    ALL = 0
    RANK_ZERO = 1


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    splitted_args = []
    for arg in args:
        assert isinstance(arg, (DataProto, DataProtoFuture))
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert isinstance(val, (DataProto, DataProtoFuture))
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


def _split_args_kwargs_data_proto_with_auto_padding(chunks, *args, **kwargs):
    from verl.protocol import DataProto, DataProtoFuture

    splitted_args = []
    splitted_kwargs = {}

    data_proto_len = None
    padding_size = None
    for arg in args:
        assert isinstance(arg, (DataProto, DataProtoFuture))
        if isinstance(arg, DataProto) and arg.is_padding_enabled():
            # for padding, we only support DataProto with same length
            if data_proto_len is None:
                data_proto_len = len(arg)
                padding_size = (
                    (chunks - (data_proto_len % chunks))
                    if (data_proto_len % chunks > 0)
                    else 0
                )
                splitted_kwargs[_padding_size_key] = padding_size
            else:
                assert data_proto_len == len(
                    arg
                ), f"expecting all arg share same length of {data_proto_len}, but got {len(arg)}"
                data_proto_len = len(arg)
            arg.padding(padding_size=padding_size)

        splitted_args.append(arg.chunk(chunks=chunks))

    for key, val in kwargs.items():
        assert isinstance(val, (DataProto, DataProtoFuture))
        if isinstance(val, DataProto) and val.is_padding_enabled():
            # for padding, we only support DataProto with same length
            if data_proto_len is None:
                data_proto_len = len(val)
                padding_size = chunks - (data_proto_len % chunks)
                splitted_kwargs[_padding_size_key] = padding_size
            else:
                assert data_proto_len == len(
                    val
                ), f"expecting all arg share same length of {data_proto_len}, but got {len(val)}"
                data_proto_len = len(val)
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(role_group, *args, **kwargs):
    args = tuple([arg] * role_group.world_size for arg in args)
    kwargs = {k: [v] * role_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dummy_direct_rollout_call(role_group, *args, **kwargs):
    raise NotImplementedError("Direct rollout call is forbidden.")


def dispatch_all_to_all(role_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(role_group, output):
    return output


def dispatch_megatron_compute(role_group, *args, **kwargs):
    """
    User passes in dp data. The data is dispatched to all tp/pp ranks with the same dp
    """

    all_args = []
    for arg in args:
        assert (
            isinstance(arg, (Tuple, List)) and len(arg) == role_group.dp_size()
        )
        transformed_args = []
        for i in range(role_group.world_size):
            local_dp_rank = role_group.get_megatron_rank_info()[i].dp_rank
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, (Tuple, List)) and len(v) == role_group.dp_size
        transformed_v = []
        for i in range(role_group.world_size):
            local_dp_rank = role_group.get_megatron_rank_info()[i].dp_rank
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_megatron_compute(role_group, output):
    """
    Only collect the data from the tp=0 and pp=last and every dp ranks
    """

    output_in_dp = []
    pp_size = role_group.get_megatron_global_info().pp_size
    for global_rank in range(role_group.world_size):
        local_rank_info = role_group.get_megatron_rank_info()[global_rank]
        if (
            local_rank_info.tp_rank == 0
            and local_rank_info.pp_rank == pp_size - 1
            and local_rank_info.cp_rank == 0
        ):
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_megatron_compute_data_proto(role_group, *args, **kwargs):
    """
    All the args and kwargs must be DataProto. The batch will be chunked by dp_size and passed to each rank
    """

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(
        role_group.dp_size(), *args, **kwargs
    )
    return dispatch_megatron_compute(
        role_group, *splitted_args, **splitted_kwargs
    )


def _concat_data_proto_or_future(output: List):
    import ray
    from verl.protocol import DataProto, DataProtoFuture

    # make sure all the elements in output has the same type
    for o in output:
        assert type(o) is type(output[0])

    o = output[0]

    if isinstance(o, DataProto):
        return DataProto.concat(output)
    elif isinstance(o, ray.ObjectRef):
        return DataProtoFuture.concat(output)
    else:
        raise NotImplementedError


def collect_megatron_compute_data_proto(role_group, output):
    """
    Each output must be a DataProto. We concat the dim=0 of output
    """
    import ray
    from verl.protocol import DataProto

    output = collect_megatron_compute(role_group, output)
    for o in output:
        assert isinstance(
            o, (DataProto, ray.ObjectRef)
        ), f"expecting {o} to be DataProto, but got {type(o)}"

    return _concat_data_proto_or_future(output)


def dispatch_megatron_pp_as_dp(role_group, *args, **kwargs):
    """
    treat pp as dp.
    """

    pp_size = role_group.pp_size()
    dp_size = role_group.dp_size()
    cp_size = role_group.cp_size()
    pp_dp_cp_size = pp_size * dp_size * cp_size

    all_args = []
    for arg in args:
        assert isinstance(arg, (List, Tuple)) and len(arg) == pp_dp_cp_size
        transformed_args = []
        for i in range(role_group.world_size):
            rank_info = role_group.get_megatron_rank_info()[i]
            local_dp_rank = rank_info.dp_rank
            local_pp_rank = rank_info.pp_rank
            local_cp_rank = rank_info.cp_rank
            # compute the rank in arg. Note that the order is dp then cp then pp
            # Also note that the outputs within a pp group will be firstly allgathered, then only the output of pp0 will be collected.
            # For pp=2 dp=4, a batch of data "ABCDEFGH" should be dispatched and collected in below order:
            #    dispatch:       pp_allgther:        collect:
            #   dp 0 1 2 3      dp  0  1  2  3
            # pp +---------+  pp +-------------+
            #  0 | A C E G |   0 | AB CD EF GH |     ABCDEFGH
            #  1 | B D F H |   1 | AB CD EF GH |
            #    +---------+     +-------------+
            dp_cp_rank = local_cp_rank * dp_size + local_dp_rank
            arg_rank = dp_cp_rank * pp_size + local_pp_rank

            transformed_args.append(arg[arg_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert (
            isinstance(v, (List, Tuple)) and len(v) == pp_dp_cp_size
        ), f"expect len(v)=={pp_dp_cp_size}, got {len(v)}"
        transformed_v = []
        for i in range(role_group.world_size):
            rank_info = role_group.get_megatron_rank_info()[i]
            local_dp_rank = rank_info.dp_rank
            local_pp_rank = rank_info.pp_rank
            local_cp_rank = rank_info.cp_rank
            # compute the rank in arg. Note that the order is dp then cp then pp
            dp_cp_rank = local_cp_rank * dp_size + local_dp_rank
            arg_rank = dp_cp_rank * pp_size + local_pp_rank
            transformed_v.append(v[arg_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_megatron_pp_as_dp(role_group, output):
    """
    treat pp as dp. Only collect data on tp=0
    """
    output_in_dp = []
    for global_rank in range(role_group.world_size):
        local_rank_info = role_group.get_megatron_rank_info()[global_rank]
        if local_rank_info.tp_rank == 0:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def collect_megatron_pp_only(role_group, output):
    """
    Only collect output of megatron pp. This is useful when examine weight names as they are identical in tp/dp
    """
    output_in_pp = []
    for global_rank in range(role_group.world_size):
        local_rank_info = role_group.get_megatron_rank_info()[global_rank]
        if local_rank_info.tp_rank == 0 and local_rank_info.dp_rank == 0:
            output_in_pp.append(output[global_rank])
    return output_in_pp


def dispatch_megatron_pp_as_dp_data_proto(role_group, *args, **kwargs):
    pp_dp_cp_size = (
        role_group.dp_size() * role_group.pp_size() * role_group.cp_size()
    )
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(
        pp_dp_cp_size, *args, **kwargs
    )
    ret = dispatch_megatron_pp_as_dp(
        role_group, *splitted_args, **splitted_kwargs
    )
    return ret


def collect_megatron_pp_as_dp_data_proto(role_group, output):
    output = collect_megatron_pp_as_dp(role_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_dp_compute(role_group, *args, **kwargs):
    for arg in args:
        assert (
            isinstance(arg, (Tuple, List))
            and len(arg) == role_group.world_size
        )
    for k, v in kwargs.items():
        assert isinstance(v, (Tuple, List)) and len(v) == role_group.world_size
    return args, kwargs


def collect_dp_compute(role_group, output):
    assert len(output) == role_group.world_size
    return output


def dispatch_dp_compute_data_proto(role_group, *args, **kwargs):
    # Note: enable auto padding for dp compute DatapProto
    (
        splitted_args,
        splitted_kwargs,
    ) = _split_args_kwargs_data_proto_with_auto_padding(
        role_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(role_group, *args, **kwargs):
    assert isinstance(
        args[0], FunctionType
    )  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(
        role_group.world_size, *args[1:], **kwargs
    )
    splitted_args_with_func = [
        [args[0]] * role_group.world_size
    ] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(role_group, output):
    import ray
    from verl.protocol import DataProto

    for o in output:
        assert isinstance(
            o, (DataProto, ray.ObjectRef)
        ), f"expecting {o} to be DataProto, but got {type(o)}"

    output = collect_dp_compute(role_group, output)
    return _concat_data_proto_or_future(output)


def get_predefined_dispatch_fn(dispatch_mode):
    predefined_dispatch_mode_fn = {
        Dispatch.ONE_TO_ALL: {
            "dispatch_fn": dispatch_one_to_all,
            "collect_fn": collect_all_to_all,
        },
        Dispatch.ALL_TO_ALL: {
            "dispatch_fn": dispatch_all_to_all,
            "collect_fn": collect_all_to_all,
        },
        Dispatch.MEGATRON_COMPUTE: {
            "dispatch_fn": dispatch_megatron_compute,
            "collect_fn": collect_megatron_compute,
        },
        Dispatch.MEGATRON_PP_AS_DP: {
            "dispatch_fn": dispatch_megatron_pp_as_dp,
            "collect_fn": collect_megatron_pp_as_dp,
        },
        Dispatch.MEGATRON_PP_ONLY: {
            "dispatch_fn": dispatch_one_to_all,
            "collect_fn": collect_megatron_pp_only,
        },
        Dispatch.MEGATRON_COMPUTE_PROTO: {
            "dispatch_fn": dispatch_megatron_compute_data_proto,
            "collect_fn": collect_megatron_compute_data_proto,
        },
        Dispatch.MEGATRON_PP_AS_DP_PROTO: {
            "dispatch_fn": dispatch_megatron_pp_as_dp_data_proto,
            "collect_fn": collect_megatron_pp_as_dp_data_proto,
        },
        Dispatch.DP_COMPUTE: {
            "dispatch_fn": dispatch_dp_compute,
            "collect_fn": collect_dp_compute,
        },
        Dispatch.DP_COMPUTE_PROTO: {
            "dispatch_fn": dispatch_dp_compute_data_proto,
            "collect_fn": collect_dp_compute_data_proto,
        },
        Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
            "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
            "collect_fn": collect_dp_compute_data_proto,
        },
        Dispatch.DP_COMPUTE_METRIC: {
            "dispatch_fn": dispatch_dp_compute_data_proto,
            "collect_fn": collect_dp_compute,
        },
        Dispatch.DIRECT_ROLLOUT_METHOD: {
            "dispatch_fn": dummy_direct_rollout_call,
            "collect_fn": dummy_direct_rollout_call,
        },
    }
    return predefined_dispatch_mode_fn[dispatch_mode]


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


def _check_dispatch_mode(dispatch_mode):
    assert isinstance(
        dispatch_mode, (Dispatch, Dict)
    ), f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    if isinstance(dispatch_mode, Dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert (
                key in dispatch_mode
            ), f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    assert isinstance(
        execute_mode, Execute
    ), f"execute_mode must be a Execute. Got {execute_mode}"


def _materialize_futures(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def register(
    dispatch_mode=Dispatch.ALL_TO_ALL,
    execute_mode=Execute.ALL,
    blocking=True,
    materialize_futures=True,
):
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return await func(*args, **kwargs)

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {
            "dispatch_mode": dispatch_mode,
            "execute_mode": execute_mode,
            "blocking": blocking,
        }
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
