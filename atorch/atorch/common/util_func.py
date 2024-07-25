import socket
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager
from operator import attrgetter
from typing import List, Union

import grpc
import torch


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]


def get_ip_address() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def grpc_server_on(
    channel,
    client_name="Unknown client",
    server_addr="unknown ip",
    timeout=60,
):
    """
    Sometimes, the client is created before the server has been created.
    If so, the client's calling to the server would raise an grpc error.
    `channel_ready_future(channel).result()` will check whether the channel
    between client and server has been established in `timeout` seconds.
    """
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise TimeoutError("{} failed to connect grpc server({}) in {}s".format(client_name, server_addr, timeout))


def wait_for_server_started(ip, port, timeout=60):
    s = socket.socket()
    num_attempts = 0
    while True:
        if num_attempts == timeout:
            raise TimeoutError("Failed to connect to {} after waiting for {} s".format((ip, port), timeout))
        try:
            s.connect((ip, port))
            break
        except socket.error:
            time.sleep(1)
            num_attempts += 1
    s.close()


def set_sync_bn_pg(model, process_group=None):
    """
    If model_context has SyncBatchNorm, set SyncBatchNorm's process group
    """
    if process_group is None:
        process_group = torch.distributed.distributed_c10d._get_default_group()
    for _, child in model.named_modules():
        if isinstance(child, torch.nn.SyncBatchNorm):
            child.process_group = process_group


def recursively_apply(
    func, data, *args, test_type=lambda t: isinstance(t, torch.Tensor), error_on_other_type=False, **kwargs
):
    if isinstance(data, Mapping):
        return type(data)(
            {
                k: recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for k, v in data.items()
            }
        )
    elif isinstance(data, (tuple, list)):
        return type(data)(
            [
                recursively_apply(
                    func, v, *args, test_type=test_type, error_on_other_type=error_on_other_type, **kwargs
                )
                for v in data
            ]
        )
    elif test_type(data):
        return func(data, *args, **kwargs)
    elif error_on_other_type:
        raise TypeError(
            f"Can't apply {func.__name__} on object of type {type(data)}, only of nested list/tuple/dicts of objects "
            f"that satisfy {test_type.__name__}."
        )
    return data


def data_to_device(data, device, non_blocking=False):
    def to_device(data, device, non_blocking=False):
        if device == "pin":
            data = data.pin_memory()
        else:
            data = data.to(device, non_blocking=non_blocking)
        return data

    return recursively_apply(to_device, data, device, non_blocking=non_blocking)


def data_float_to_dtype(inputs, dtype):
    if isinstance(inputs, (list, tuple)):
        new_inputs = []
        for v in inputs:
            new_inputs.append(data_float_to_dtype(v, dtype))
        return inputs.__class__(new_inputs)
    elif isinstance(inputs, dict):
        new_inputs = {}
        for k, v in inputs.items():
            new_inputs[k] = data_float_to_dtype(v, dtype)
        return new_inputs
    elif (
        isinstance(inputs, torch.Tensor)
        and inputs.dtype != dtype
        and inputs.dtype in (torch.float32, torch.half, torch.bfloat16)
    ):
        return inputs.to(dtype)
    else:
        return inputs


def check_and_transform_to_list(tensor: Union[List, torch.Tensor]):
    if not isinstance(tensor, list):
        tensor = [tensor]
    return tensor


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def recursive_setattr(obj, attr, value):
    if "." not in attr:
        setattr(obj, attr, value)
    else:
        dst_split = attr.split(".")
        dst_name = dst_split.pop()
        dst_module_name = ".".join(dst_split)
        dst_module = attrgetter(dst_module_name)(obj)
        setattr(dst_module, dst_name, value)


def is_wrapped_by_context_manager(func):
    closure = func.__closure__
    if closure is None:
        return False

    for cell in closure:
        if isinstance(cell.cell_contents, AbstractContextManager):
            return True

    return False
