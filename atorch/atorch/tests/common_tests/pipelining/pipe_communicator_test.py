import os
import unittest

import torch
import torch.multiprocessing as mp

import atorch
from atorch.common.util_func import find_free_port
from atorch.communication.pipe_communicator import PipeCommunicator, RecvBuffer
from atorch.distributed.distributed import create_parallel_group, pipe_next_rank, pipe_prev_rank


def _create_pipe_group(rank):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    set_cuda_device = backend == "nccl"
    res = atorch.init_distributed(backend, set_cuda_device_using_local_rank=set_cuda_device)
    assert res
    world_size = torch.distributed.get_world_size()
    gpu_partition = ([("pipe", world_size)], None)
    create_parallel_group(gpu_partition, use_atorch_pipe=True)


def _check_cached_meta(tensors, pipe_communicator, meta_id, tensor_num):
    ndim_sum = 0
    meta = pipe_communicator._meta_cache[meta_id][1]
    for i in range(tensor_num):
        ndim = meta[i * 3]
        ndim_sum += ndim
        dtype = pipe_communicator._int_to_dtype[meta[i * 3 + 1]]
        requires_grad = bool(meta[i * 3 + 2])
        tensor_shape = meta[tensor_num * 3 + ndim_sum - ndim : tensor_num * 3 + ndim_sum]
        assert tensors[i].ndim == ndim
        assert tensors[i].dtype == dtype
        assert tensors[i].requires_grad == requires_grad
        assert tensors[i].size() == torch.Size(tensor_shape)


def _check_cached_buffer(tensors, meta_id, tensor_num):
    buffer = RecvBuffer().get(meta_id)
    assert len(buffer) == len(tensors)
    for i in range(tensor_num):
        assert buffer[i].shape == tensors[i].shape
        assert buffer[i].dtype == tensors[i].dtype
        assert buffer[i].requires_grad == tensors[i].requires_grad
        assert torch.allclose(buffer[i], tensors[i])


def pipe_send_or_recv_fn(rank):
    _create_pipe_group(rank)

    pipe_communicator = PipeCommunicator()
    tensor_num = 5
    tensor_shapes = [torch.Size((2, 2, 3))] * 2 + [torch.Size((4, 5))] * 3
    tensors = [torch.zeros(tensor_shapes[i], dtype=torch.float32, device="cuda") for i in range(tensor_num)]
    tensors[0].requires_grad = False
    tensors[1].requires_grad = True

    if rank == 0:
        meta_id = "send_meta"
        pipe_communicator.send(tensors, target=pipe_next_rank(), meta_id=meta_id)

        assert meta_id in pipe_communicator._meta_cache
        assert tensor_num == pipe_communicator._meta_cache[meta_id][0]
        _check_cached_meta(tensors, pipe_communicator, meta_id, tensor_num)

    elif rank == 1:
        meta_id = "recv_meta"
        recv_tensors = pipe_communicator.recv(src=pipe_prev_rank(), meta_id=meta_id)
        assert meta_id in pipe_communicator._meta_cache
        assert tensor_num == pipe_communicator._meta_cache[meta_id][0]

        for i, tensor in enumerate(recv_tensors):
            assert tensor_shapes[i] == tensor.size()
            assert torch.allclose(tensors[i], tensor)
            assert tensors[i].dtype == tensor.dtype
            assert tensors[i].requires_grad == tensor.requires_grad

        _check_cached_meta(tensors, pipe_communicator, meta_id, tensor_num)
        _check_cached_buffer(tensors, meta_id, tensor_num)

    atorch.reset_distributed()


def _exec_batch_send(pipe_communicator, tensors, meta_id, tensor_num):
    send_ops = pipe_communicator.send_ops_to_batch(tensors, target=pipe_next_rank(), meta_id=meta_id)
    send_work = pipe_communicator.batch_p2p(send_ops)

    if send_work:
        send_work.wait()

    assert meta_id in pipe_communicator._meta_cache
    assert tensor_num == pipe_communicator._meta_cache[meta_id][0]


def _exec_batch_recv(pipe_communicator, meta_id, tensor_num):
    recv_ops, recv_tensors = pipe_communicator.recv_ops_to_batch(src=pipe_prev_rank(), meta_id=meta_id)
    recv_work = pipe_communicator.batch_p2p(recv_ops)

    if recv_work:
        recv_work.wait()

    assert meta_id in pipe_communicator._meta_cache
    assert tensor_num == pipe_communicator._meta_cache[meta_id][0]

    return recv_tensors


def batch_send_or_recv_fn(rank):
    _create_pipe_group(rank)

    pipe_communicator = PipeCommunicator()
    tensor_num = 5
    tensor_shapes = [torch.Size((2, 2, 3))] * 2 + [torch.Size((4, 5))] * 3
    tensors = [torch.zeros(tensor_shapes[i], dtype=torch.float32, device="cuda") for i in range(tensor_num)]
    tensors[0].requires_grad = False
    tensors[1].requires_grad = True

    if rank == 0:
        meta_id = "send_meta"
        _exec_batch_send(pipe_communicator, tensors, meta_id, tensor_num)
        _check_cached_meta(tensors, pipe_communicator, meta_id, tensor_num)

        _exec_batch_send(pipe_communicator, tensors, meta_id, tensor_num)

    elif rank == 1:
        meta_id = "recv_meta"
        recv_tensors = _exec_batch_recv(pipe_communicator, meta_id, tensor_num)

        for i, tensor in enumerate(recv_tensors):
            assert tensor_shapes[i] == tensor.size()
            assert torch.allclose(tensors[i], tensor)
            assert tensors[i].dtype == tensor.dtype
            assert tensors[i].requires_grad == tensor.requires_grad

        _check_cached_meta(tensors, pipe_communicator, meta_id, tensor_num)
        _check_cached_buffer(tensors, meta_id, tensor_num)

        _exec_batch_recv(pipe_communicator, meta_id, tensor_num)

    atorch.reset_distributed()


def cross_batch_send_or_recv_fn(rank):
    _create_pipe_group(rank)

    pipe_communicator = PipeCommunicator()
    tensor_num = 5

    tensor_shapes_1 = [torch.Size((2, 2, 3))] * 2 + [torch.Size((4, 5))] * 3
    tensors1 = [torch.zeros(tensor_shapes_1[i], dtype=torch.float32, device="cuda") for i in range(tensor_num)]
    tensors1[0].requires_grad = False
    tensors1[1].requires_grad = True

    tensor_shapes_2 = [torch.Size((4, 3))] * 3 + [torch.Size((2, 3, 2))] * 2
    tensors2 = [torch.zeros(tensor_shapes_2[i], dtype=torch.float32, device="cuda") for i in range(tensor_num)]
    tensors2[0].requires_grad = True
    tensors2[1].requires_grad = False

    if rank == 0:
        meta_id = "send_meta_rank0"
        send_ops = pipe_communicator.send_ops_to_batch(tensors1, target=pipe_next_rank(), meta_id=meta_id)
        send_work = pipe_communicator.batch_p2p(send_ops)

        meta_id = "recv_meta_rank0"
        recv_ops, recv_tensors2 = pipe_communicator.recv_ops_to_batch(src=pipe_prev_rank(), meta_id=meta_id)
        recv_work = pipe_communicator.batch_p2p(recv_ops)

        if recv_work:
            recv_work.wait()

        assert meta_id in pipe_communicator._meta_cache
        assert tensor_num == pipe_communicator._meta_cache[meta_id][0]

        for i, tensor in enumerate(recv_tensors2):
            assert tensor_shapes_2[i] == tensor.size()
            assert torch.allclose(tensors2[i], tensor)
            assert tensors2[i].dtype == tensor.dtype
            assert tensors2[i].requires_grad == tensor.requires_grad

        _check_cached_meta(tensors2, pipe_communicator, meta_id, tensor_num)

        if send_work:
            send_work.wait()

        meta_id = "send_meta_rank0"
        assert meta_id in pipe_communicator._meta_cache
        assert tensor_num == pipe_communicator._meta_cache[meta_id][0]
        _check_cached_meta(tensors1, pipe_communicator, meta_id, tensor_num)

    elif rank == 1:
        meta_id = "recv_meta_rank1"
        recv_ops, recv_tensors1 = pipe_communicator.recv_ops_to_batch(src=pipe_prev_rank(), meta_id=meta_id)
        recv_work = pipe_communicator.batch_p2p(recv_ops)

        if recv_work:
            recv_work.wait()

        assert meta_id in pipe_communicator._meta_cache
        assert tensor_num == pipe_communicator._meta_cache[meta_id][0]

        for i, tensor in enumerate(recv_tensors1):
            assert tensor_shapes_1[i] == tensor.size()
            assert torch.allclose(tensors1[i], tensor)
            assert tensors1[i].dtype == tensor.dtype
            assert tensors1[i].requires_grad == tensor.requires_grad

        _check_cached_meta(tensors1, pipe_communicator, meta_id, tensor_num)

        meta_id = "send_meta_rank1"
        send_ops = pipe_communicator.send_ops_to_batch(tensors2, target=pipe_prev_rank(), meta_id=meta_id)
        send_work = pipe_communicator.batch_p2p(send_ops)

        if send_work:
            send_work.wait()

        assert meta_id in pipe_communicator._meta_cache
        assert tensor_num == pipe_communicator._meta_cache[meta_id][0]
        _check_cached_meta(tensors2, pipe_communicator, meta_id, tensor_num)

    atorch.reset_distributed()


class PipeCommunicatorTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires 2 gpus.")
    def test_send_recv(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            pipe_send_or_recv_fn,
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires 2 gpus.")
    def test_batch_send_recv(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            batch_send_or_recv_fn,
            nprocs=world_size,
            join=True,
        )

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires 2 gpus.")
    def test_cross_batch_send_recv(self):
        world_size = 2
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["WORLD_SIZE"] = str(world_size)
        mp.spawn(
            cross_batch_send_or_recv_fn,
            nprocs=world_size,
            join=True,
        )
