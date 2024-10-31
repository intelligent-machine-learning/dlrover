from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import ProcessGroup

from atorch.common.singleton import SingletonMeta
from atorch.common.util_func import check_and_transform_to_list
from atorch.distributed import distributed as _distributed_context


class RecvBuffer(metaclass=SingletonMeta):
    _buffer: Dict = {}

    def exist(self, meta_id: str):
        return meta_id in self._buffer

    def put(self, meta_id: str, buffer: Any):
        self._buffer[meta_id] = buffer

    def get(self, meta_id: str):
        return self._buffer[meta_id]


def _broadcast_tensor_list(
    tensor_list: List[torch.Tensor],
    src: int,
    group: ProcessGroup,
    device: Optional[Union[torch.device, str, int]] = None,
):
    pass


def _communicate_metas(meta, recv_src_meta_len, src_rank, target_rank):
    """Communicate tensor metas.

    Args:
        mata: meta value to send(no tensor sent if
                          set to None).
        recv_src_meta_len: meta tensor length to recv(not recv if set to None).
        src_rank: recv meta from src rank.
        target_rank: send meta to target rank.
    Returns:
        Optional, (recv_src_shape, )
    """

    recv_src_meta_tensor = None
    send_target_meta_tensor = None
    if recv_src_meta_len is not None:
        recv_src_meta_tensor = torch.empty((recv_src_meta_len), device=torch.cuda.current_device(), dtype=torch.int64)
    if meta is not None:
        send_target_meta_tensor = torch.tensor(meta, device=torch.cuda.current_device(), dtype=torch.int64)

    ops = []
    if recv_src_meta_tensor is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            recv_src_meta_tensor,
            src_rank,
        )
        ops.append(recv_prev_op)
    if send_target_meta_tensor is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            send_target_meta_tensor,
            target_rank,
        )
        ops.append(send_next_op)

    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # To protect against race condition when using batch_isend_irecv().
    # should take this out once the bug with batch_isend_irecv is resolved.
    torch.cuda.synchronize()

    if recv_src_meta_tensor is not None:
        recv_src_meta = recv_src_meta_tensor.tolist()

        return recv_src_meta


def _p2p_ops(
    *,
    tensor_recv_src: Optional[torch.Tensor],
    tensor_send_target: Optional[torch.Tensor],
    src_rank: int,
    target_rank: int,
    group: torch.distributed.ProcessGroup
):
    """
    p2p operator

    Args:
        tensor_recv_src: torch tensor to receive data from src rank
        tensor_send_target: torch tensor to send
        src_rank: source rank which irecv op receive data from
        target_rank: target rank which isend op send data to

    Return:
        list of reqs
    """
    reqs = []
    if _distributed_context.parallel_rank("pipe") % 2 == 0:
        if tensor_send_target is not None:
            send_target_req = torch.distributed.isend(
                tensor=tensor_send_target,
                dst=target_rank,
                group=group,
            )
            reqs.append(send_target_req)

        if tensor_recv_src is not None:
            recv_src_req = torch.distributed.irecv(
                tensor=tensor_recv_src,
                src=src_rank,
                group=group,
            )
            reqs.append(recv_src_req)
    else:
        if tensor_recv_src is not None:
            recv_src_req = torch.distributed.irecv(
                tensor=tensor_recv_src,
                src=src_rank,
                group=group,
            )
            reqs.append(recv_src_req)

        if tensor_send_target is not None:
            send_target_req = torch.distributed.isend(
                tensor=tensor_send_target,
                dst=target_rank,
                group=group,
            )
            reqs.append(send_target_req)

    return reqs


def _p2p_op_to_batch(
    tensor_recv_src: Optional[torch.Tensor],
    tensor_send_target: Optional[torch.Tensor],
    src_rank: int,
    target_rank: int,
    group: torch.distributed.ProcessGroup,
):
    """
    P2POp instance to batch send/recv
    """
    assert tensor_recv_src is not None or tensor_send_target is not None
    if tensor_send_target is not None:
        return torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_target,
            target_rank,
            group,
        )

    if tensor_recv_src is not None:
        return torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_src,
            src_rank,
            group,
        )


def _p2p_communicate(
    *,
    tensor_send_target: Optional[torch.Tensor],
    recv_src: bool,
    tensor_shape: Tuple,
    pipe_dtype: torch.dtype,
    src_rank: int,
    target_rank: int,
    group: torch.distributed.ProcessGroup,
    wait_on_reqs: bool = True,
    use_ring_exchange_p2p: bool = False,
    batch_p2p_comm: bool = False,
    batch_p2p_sync: bool = False,
    buffer=None
):
    """
    execute p2p communication, support single op just.
    """
    recv_src_shape = tensor_shape
    tensor_recv_src = None

    if recv_src:
        if buffer is not None:
            tensor_recv_src = buffer
        else:
            tensor_recv_src = torch.empty(
                recv_src_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=pipe_dtype,
            )

    if use_ring_exchange_p2p:
        pass
    elif batch_p2p_comm:
        # Note: this `batch` is to batch recv and send ops
        assert wait_on_reqs
        pass
    else:
        p2p_op = _p2p_ops

    reqs = p2p_op(
        tensor_recv_src=tensor_recv_src,
        tensor_send_target=tensor_send_target,
        src_rank=src_rank,
        target_rank=target_rank,
        group=group,
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if batch_p2p_comm and batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recv_src


def _batch_p2p_communicate_ops(tensors, recv_src, tuple_metas, src_rank, target_rank, group, buffer=None):
    """
    batch multi send ops or multi recv ops
    """
    ops = []

    if recv_src:
        received_tensors = []
        ndim_sum = 0
        tensor_num, metas = tuple_metas[0], tuple_metas[1]
        if buffer is not None:
            assert tensor_num == len(buffer)

        for i in range(tensor_num):
            ndim = metas[i * 3]
            ndim_sum += ndim
            pipe_dtype = PipeCommunicator._int_to_dtype[metas[i * 3 + 1]]
            requires_grad = bool(metas[i * 3 + 2])
            recv_src_shape = metas[tensor_num * 3 + ndim_sum - ndim : tensor_num * 3 + ndim_sum]

            if buffer is not None:
                tensor_recv_src = buffer[i]
                assert (
                    pipe_dtype == tensor_recv_src.dtype
                    and requires_grad == tensor_recv_src.requires_grad
                    and recv_src_shape == list(tensor_recv_src.shape)
                )
            else:
                tensor_recv_src = torch.empty(
                    recv_src_shape,
                    requires_grad=requires_grad,
                    device=torch.cuda.current_device(),
                    dtype=pipe_dtype,
                )
            recv_op = _p2p_op_to_batch(tensor_recv_src, None, src_rank, None, group)
            ops.append(recv_op)
            received_tensors.append(tensor_recv_src)

        return ops, received_tensors
    elif tensors is not None:
        for tensor_send_target in tensors:
            send_op = _p2p_op_to_batch(None, tensor_send_target, None, target_rank, group)
            ops.append(send_op)

        return ops


def _init_dtypes(invert=False):
    torch_attrs = dir(torch)
    dtypes = [getattr(torch, attr) for attr in torch_attrs if isinstance(getattr(torch, attr), torch.dtype)]

    if invert:
        return {dtype: i for i, dtype in enumerate(dtypes)}
    return {i: dtype for i, dtype in enumerate(dtypes)}


class PipeCommunicator:
    _meta_cache: Dict = {}
    _int_to_dtype: Dict = _init_dtypes()
    _dtype_to_int: Dict = _init_dtypes(invert=True)

    def _send_tensor_metas(
        self, tensors: List[torch.Tensor], target: int, meta_id: Optional[str] = None, ignore_old_meta: bool = False
    ):
        tensor_num = len(tensors)
        total_shape_dim = sum([tensor.ndim for tensor in tensors])
        if meta_id is None or ignore_old_meta or meta_id not in self._meta_cache:
            # Communicate meta
            # communicate meta of meta, tensor num and total shape dim
            meta_of_meta = [tensor_num, total_shape_dim]
            _communicate_metas(meta=meta_of_meta, recv_src_meta_len=None, src_rank=None, target_rank=target)

            # communicate meta
            metas = []
            for tensor in tensors:
                metas.extend([tensor.ndim, self._dtype_to_int[tensor.dtype], int(tensor.requires_grad)])
            for tensor in tensors:
                metas.extend(list(tensor.size()))
            _communicate_metas(meta=metas, recv_src_meta_len=None, src_rank=None, target_rank=target)

            tuple_metas = (tensor_num, metas)
            if meta_id is not None:
                self._meta_cache[meta_id] = tuple_metas
        else:
            tuple_metas = self._meta_cache[meta_id]
        assert tensor_num == tuple_metas[0]
        assert len(tuple_metas[1]) == (tensor_num * 3 + total_shape_dim)

    def _recv_tensor_metas(self, src: int, meta_id: Optional[str] = None, ignore_old_meta: bool = False):
        if meta_id is None or ignore_old_meta or meta_id not in self._meta_cache:
            # Communicate meta
            # communicate meta of meta, tensor num and total shape dim
            meta_of_meta = _communicate_metas(meta=None, recv_src_meta_len=2, src_rank=src, target_rank=None)
            tensor_num, total_shape_dim = meta_of_meta[0], meta_of_meta[1]

            # communicate meta
            recv_src_meta_len = tensor_num * 3 + total_shape_dim
            metas = _communicate_metas(
                meta=None,
                recv_src_meta_len=recv_src_meta_len,
                src_rank=src,
                target_rank=None,
            )
            assert len(metas) == recv_src_meta_len

            tuple_metas = (tensor_num, metas)
            if meta_id is not None:
                self._meta_cache[meta_id] = tuple_metas
        else:
            tuple_metas = self._meta_cache[meta_id]

        return tuple_metas

    def get_meta(self, meta_id: str):
        return self._meta_cache[meta_id]

    def send(
        self,
        tensor: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        target: int,
        meta_id: Optional[str] = None,
        ignore_old_meta: bool = False,
    ):
        """
        send tensor or tensors one by one.

        Args:
            tensor: tensor or tensor list to send
            target: target rank
            meta_id: string id to record or load meta
            ignore_old_meta: ignore old meta forcely
        """
        tensors = check_and_transform_to_list(tensor)
        self._send_tensor_metas(tensors, target, meta_id, ignore_old_meta)

        for tensor in tensors:
            _p2p_communicate(
                tensor_send_target=tensor,
                recv_src=False,
                tensor_shape=None,
                pipe_dtype=None,
                src_rank=None,
                target_rank=target,
                group=_distributed_context.parallel_group("pipe"),
            )

    def recv(self, src, meta_id=None, ignore_old_meta=False):
        """
        recv tensor one by one

        Args:
            src: src rank to recv tensor from
            meta_id: string id to record or load meta
            ignore_old_meta: ignore old meta forcely

        Return:
            List[torch.Tensor]
        """
        tuple_metas = self._recv_tensor_metas(src, meta_id, ignore_old_meta)
        tensor_num, metas = tuple_metas[0], tuple_metas[1]

        if meta_id is None or ignore_old_meta or not RecvBuffer().exist(meta_id):
            buffer = None
        else:
            buffer = RecvBuffer().get(meta_id)
            assert tensor_num == len(buffer)

        tensors = []
        ndim_sum = 0
        for i in range(tensor_num):
            ndim = metas[i * 3]
            ndim_sum += ndim
            dtype = self._int_to_dtype[metas[i * 3 + 1]]
            requires_grad = bool(metas[i * 3 + 2])
            tensor_shape = metas[tensor_num * 3 + ndim_sum - ndim : tensor_num * 3 + ndim_sum]
            tensor = _p2p_communicate(
                tensor_send_target=None,
                recv_src=True,
                tensor_shape=tensor_shape,
                pipe_dtype=dtype,
                src_rank=src,
                target_rank=None,
                group=_distributed_context.parallel_group("pipe"),
                buffer=buffer[i] if buffer is not None else buffer,
            )
            tensor.requires_grad = requires_grad
            assert tensor.dtype == dtype and list(tensor.shape) == tensor_shape
            tensors.append(tensor)

        if meta_id is not None and (ignore_old_meta or not RecvBuffer().exist(meta_id)):
            RecvBuffer().put(meta_id, tensors)

        return tensors

    def send_ops_to_batch(
        self,
        tensor: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        target: int,
        meta_id: Optional[str] = None,
        ignore_old_meta: bool = False,
    ):
        """
        ops to batch send

        Args:
            tensor: tensor or tensor list to send
            target: target rank
            meta_id: string id to record or load meta
            ignore_old_meta: ignore old meta forcely

        Return:
            List[torch.distributed.P2POp]
        """
        tensors = check_and_transform_to_list(tensor)
        self._send_tensor_metas(tensors, target, meta_id, ignore_old_meta)

        return _batch_p2p_communicate_ops(
            tensors, False, None, None, target, _distributed_context.parallel_group("pipe")
        )

    def recv_ops_to_batch(self, src, meta_id=None, ignore_old_meta=False):
        """
        ops to batch recv

        Args:
            src: src rank to recv tensor from
            meta_id: string id to record or load meta
            ignore_old_meta: ignore old meta forcely

        Return:
            List[torch.distributed.P2POp], List[torch.Tensor]
        """
        tuple_metas = self._recv_tensor_metas(src, meta_id, ignore_old_meta)
        if meta_id is None or ignore_old_meta or not RecvBuffer().exist(meta_id):
            buffer = None
        else:
            buffer = RecvBuffer().get(meta_id)

        ops, received_tensors = _batch_p2p_communicate_ops(
            None, True, tuple_metas, src, None, _distributed_context.parallel_group("pipe"), buffer
        )

        if meta_id is not None and (ignore_old_meta or not RecvBuffer().exist(meta_id)):
            RecvBuffer().put(meta_id, received_tensors)

        return ops, received_tensors

    def batch_p2p(self, p2p_ops: List[torch.distributed.P2POp]):
        """
        batch send/recv ops
        """
        if len(p2p_ops) == 0:
            return None

        return torch.distributed.batch_isend_irecv(p2p_ops).pop()
