# Forked from PiPPy. Maintaining a local copy of PipelineStage.
# Hacks the original impl properly so it runs for our models.
import inspect
import logging
import operator
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

try:
    import pippy
    from pippy.backward import stage_backward
    from pippy.debug import map_debug_info
    from pippy.fx import Node
    from pippy.fx.passes.shape_prop import TensorMetadata
    from pippy.IR import Pipe
    from pippy.microbatch import merge_chunks, split_args_kwargs_into_chunks
    from pippy.utils import flatten_args
except ImportError:
    (
        pippy,
        stage_backward,
        map_debug_info,
        TensorMetadata,
        Node,
        Pipe,
        merge_chunks,
        split_args_kwargs_into_chunks,
        flatten_args,
    ) = [None] * 9


def _make_tensor_from_meta(
    tensor_meta: TensorMetadata,
    device: torch.device,
) -> torch.Tensor:
    return torch.empty(tensor_meta.shape, dtype=tensor_meta.dtype, device=device)


def compute_tag(
    is_grad: bool, source_stage_id: int, receiver_stage_id: int, chunk: int, chunks: int, num_stages: int
) -> int:
    if not is_grad:
        message_type_identifier = 0
    else:
        message_type_identifier = 1

    tag = 0
    tag += message_type_identifier * num_stages**3 * chunks
    tag += source_stage_id * num_stages**2 * chunks
    tag += receiver_stage_id * chunks * num_stages
    tag += chunk
    return tag


class ReadWork:
    def __init__(self, read_done_condition, key, read_done_flag, cache_lock, cache):
        self.read_done_condition = read_done_condition
        self.key = key
        self.read_done_flag = read_done_flag
        self.cache_lock = cache_lock
        self.cache = cache

    def wait(self):
        with self.read_done_condition:
            while not self.read_done_flag[self.key]:
                self.read_done_condition.wait()

        with self.cache_lock:
            # Decrease the reference count
            tensor, ref_count = self.cache[self.key]
            ref_count -= 1
            # If no more references, remove the tensor from the cache
            if ref_count == 0:
                del self.cache[self.key]
            else:
                self.cache[self.key] = (tensor, ref_count)


class SendWork:
    def __init__(self, write_done_condition, key, write_done_flag):
        self.write_done_condition = write_done_condition
        self.key = key
        self.write_done_flag = write_done_flag

    def wait(self):
        with self.write_done_condition:
            while not self.write_done_flag[self.key]:
                self.write_done_condition.wait()


class RecvInfo:
    def __init__(
        self,
        input_name: str,
        source: int,
        buffer: torch.Tensor,
    ):
        self.input_name = input_name
        self.source = source
        self.buffer = buffer

    def __repr__(self):
        return f"RecvInfo(input={self.input_name}, source={self.source}, buffer={self.buffer.size()})"


class StageArgPlaceholder:
    pass


class PipelineStage(torch.nn.Module):
    def __init__(
        self,
        pipe: Pipe,
        stage_index: int,
        nstages: int,
        chunks: int,
        device: torch.device,
        checkpoint: bool = False,
        group: dist.ProcessGroup = None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
        forward_keys=None,
    ):
        super().__init__()
        self.pipe = pipe
        self.stage_index = stage_index
        self.nstages = nstages
        self.chunks = chunks
        self.device = device
        self.checkpoint = checkpoint
        self.group = group
        self.args_chunk_spec = args_chunk_spec
        self.kwargs_chunk_spec = kwargs_chunk_spec
        self.output_chunk_spec = output_chunk_spec

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(group)

        # Run time states
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: Dict[int, Any] = {}
        # Split input chunks
        self.args_split = None
        self.kwargs_split = None
        # Activation send requests of all chunk
        self.all_act_send_reqs: List[dist.Work] = []
        # Grad send requests of all chunk
        self.all_grad_send_reqs: List[dist.Work] = []
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: List[Any] = []

        self.read_done_flags: Dict[int, Dict[Tuple, bool]] = {chunk: {} for chunk in range(self.chunks)}
        self.write_done_flags: Dict[int, Dict[Tuple, bool]] = {chunk: {} for chunk in range(self.chunks)}

        # Find my submodule
        self.split_gm = self.pipe.split_gm
        named_children = list(self.split_gm.named_children())
        self.name, self.submod = named_children[stage_index]
        logging.info(
            f"[{self.stage_index}][{self.name}] "
            f"[{self.group_rank}][{self.name}] "
            f"Creating PipelineStage:\n"
            f"{self.submod}"
        )

        # Hack to sieve the inputs
        self.forward_keys = forward_keys
        self.submod_forward_keys = list(node.target for node in self.submod.graph.nodes if node.op == "placeholder")

        # Find my forward node in graph
        found_node = False
        for node in self.split_gm.graph.nodes:
            if node.name == self.name:
                self.node = node
                found_node = True
                break
        if not found_node:
            raise AssertionError(f"Cannot find {self.name} in graph")

        # Find my backward node in graph
        if self.pipe.has_loss_and_backwards:
            found_bwd = False
            seen_bwd = -1
            for node in reversed(self.split_gm.graph.nodes):
                if (node.op, node.target) == ("call_function", stage_backward):
                    seen_bwd += 1
                    if seen_bwd == self.stage_index:
                        found_bwd = True
                        self.bwd_node = node
                        break
            if not found_bwd:
                raise AssertionError(f"Cannot find backward for {self.name} in graph")

        # Create submod to rank mapping
        self.submod_to_stage_index: Dict[str, int] = {}
        for i, (name, _) in enumerate(self.split_gm.named_children()):
            self.submod_to_stage_index.setdefault(name, i)

        # Create stage id to group rank mapping
        # In interleaved case, `group_rank` is stage index % group size.
        self.stage_index_to_group_rank: Dict[int, int] = {}
        pg_world_size = dist.get_world_size(group)
        for i in range(nstages):
            # We only support wrapped-around interleaving
            peer_rank = i % pg_world_size
            self.stage_index_to_group_rank.setdefault(i, peer_rank)

        # Prepare send/recv infrastructure
        self._prepare_send_recv_infra()

    def set_cache(self, caches, cache_locks, wait_conditions):
        self.caches = caches
        self.cache_locks = cache_locks
        self.wait_conditions = wait_conditions

    def is_first(self):
        return self.stage_index == 0

    def is_last(self):
        return self.stage_index == self.nstages - 1

    def _prepare_send_recv_infra(self):
        """
        Create send/recv infrastructures for activations (during forward) and
        gradients (during backward)
        """
        # chunk : Tuple of arg buffers
        self.args_recv_info: Dict[int, Tuple] = {}
        # chunk : Dict of kwarg buffers
        self.kwargs_recv_info: Dict[int, Dict] = {}
        for chunk in range(self.chunks):
            (
                self.args_recv_info[chunk],
                self.kwargs_recv_info[chunk],
            ) = self._create_act_recv_buffers()

        # Send info during forward for each activation
        self.act_send_info = self._create_act_send_info()

        if self.pipe.has_loss_and_backwards:
            # chunk : List of output grad buffers
            # `grad_recv_info` is a mirror of `act_send_info`
            self.grad_recv_info: Dict = {}
            for chunk in range(self.chunks):
                self.grad_recv_info[chunk] = self._create_grad_recv_info(self.act_send_info)

            # Send info for input grads during backward
            # List of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info` + `kwargs_recv_info`
            self.grad_send_info = self._create_grad_send_info(
                self.args_recv_info[0],
                self.kwargs_recv_info[0],
            )

    def get_stage_index_of_submod(
        self,
        submod_name: str,
    ):
        if submod_name not in self.submod_to_stage_index:
            raise AssertionError(f"Stage id of {submod_name} not found")

        return self.submod_to_stage_index[submod_name]

    def _create_act_recv_buffers(
        self,
    ):
        def create_recv_tensor(
            input_node,
            output_idx: Optional[int] = None,
        ):
            """
            Create a tensor for receiving the `output_idx`-th value from
            `input_node`
            """
            if input_node.op == "placeholder":
                # Do not create buffer for placeholder
                return StageArgPlaceholder()

            # In case the input is a `getitem` node, we recursively find the
            # real source e.g. getitem1 = submod0[1]
            # Here `submod0` is args[0], 1 is args[1]
            if input_node.target is operator.getitem:
                if "tensor_meta" in input_node.meta:
                    real_input_node = input_node.args[0]
                    out_idx = input_node.args[1]
                    return create_recv_tensor(real_input_node, out_idx)
                else:
                    raise NotImplementedError(
                        f"getitem gets a non-Tensor value, this is not yet supported. "
                        f"Node: {input_node.format_node()}"
                    )

            if output_idx is not None:
                # If a node has multiple output values, "tensor_meta" is a list
                # of tensor meta
                tensor_meta = input_node.meta["tensor_meta"][output_idx]
            else:
                tensor_meta = input_node.meta["tensor_meta"]

            logging.info(
                f"[{self.group_rank}][{self.name}] "
                f"Creating recv buffer for input '{input_node.name}' "
                f"value index {output_idx}: {tensor_meta.shape}"
            )

            src_rank = self.get_stage_index_of_submod(input_node.name)
            buffer = _make_tensor_from_meta(tensor_meta, self.device)
            # Enable gradient in training mode
            if self.pipe.has_loss_and_backwards and tensor_meta.requires_grad:
                buffer.requires_grad_(True)
            return RecvInfo(
                input_node.name,
                src_rank,
                buffer,
            )

        # `args` is a Tuple, hence we will have:
        # Tuple[RecvInfo]
        args_recv_info = pippy.fx.node.map_arg(self.node.args, create_recv_tensor)

        # `kwargs` is a Dict, hence we will have:
        # Dict[keyword, RecvInfo]
        kwargs_recv_info = pippy.fx.node.map_arg(self.node.kwargs, create_recv_tensor)

        logging.info(f"[{self.group_rank}][{self.name}] " f"Activation recv info: {args_recv_info}")
        return args_recv_info, kwargs_recv_info

    def find_dst_rank(
        self,
        user: Node,
    ) -> Optional[int]:
        """
        Find the destination rank of a `user` node.
        If the `user` is not a submod, `None` may be returned.
        """
        if user.op == "call_module":
            # User is a stage (`call_module`)
            return self.get_stage_index_of_submod(user.name)
        # elif user.target is sync_barrier:
        #     # Send result back to pp rank 0
        #     return 0
        else:
            # - If user.op == "output":
            #   No need to send back to rank 0
            # - If user.target is stage_backward:
            #   No need to send assuming submod output is stored locally or
            #   should be re-calucated in case of activation checkpointing
            return None

    def _create_act_send_info(self):
        # Output index: List of receiver ranks
        act_send_info: Dict[int, List] = {}
        out_idx = 0

        for user in self.node.users:
            if user.target is operator.getitem:
                # Recursively find the real destination
                gi_dsts = act_send_info.setdefault(out_idx, [])
                for gi_user in user.users:
                    dst_rank = self.find_dst_rank(gi_user)
                    if dst_rank is not None:
                        gi_dsts.append(dst_rank)
                # Next `getitem` will point to the next output index
                out_idx += 1
            else:
                # In case of single output value, `out_idx` will not increase
                dsts = act_send_info.setdefault(out_idx, [])
                dst_rank = self.find_dst_rank(user)
                if dst_rank is not None:
                    dsts.append(dst_rank)

        logging.info(f"[{self.group_rank}][{self.name}] " f"Send info: {act_send_info}")
        return act_send_info

    def _create_grad_recv_info(
        self,
        act_send_info: Dict,
    ) -> Dict[int, RecvInfo]:
        # Dict[output_index, RecvInfo]
        grad_recv_info: Dict = defaultdict(list)
        my_tensor_meta = self.node.meta["tensor_meta"]

        for out_idx, dst_list in act_send_info.items():
            if not dst_list:
                # No actual receiver for activation so no grad coming back
                continue

            # TODO: clean way
            if len(act_send_info) > 1:
                tensor_meta = my_tensor_meta[out_idx]
            else:
                tensor_meta = my_tensor_meta

            if not tensor_meta.requires_grad:
                continue

            for grad_src in dst_list:
                grad_recv_info[out_idx].append(
                    RecvInfo(
                        f"{grad_src}",
                        grad_src,
                        _make_tensor_from_meta(tensor_meta, self.device),
                    )
                )

        logging.info(f"[{self.group_rank}][{self.name}] " f"Grad recv info: {grad_recv_info}")
        return grad_recv_info

    def _create_grad_send_info(
        self,
        args_recv_info: Tuple,
        kwargs_recv_info: Dict,
    ) -> List[Optional[int]]:
        grad_send_info: List[Optional[int]] = []

        def map_recv_to_send(a):
            if isinstance(a, RecvInfo) and a.buffer.requires_grad:
                grad_send_info.append(a.source)
                return a.source
            else:
                grad_send_info.append(None)
                return None

        pippy.fx.node.map_aggregate(args_recv_info, map_recv_to_send)

        pippy.fx.node.map_aggregate(kwargs_recv_info, map_recv_to_send)

        logging.info(f"[{self.group_rank}][{self.name}] " f"Grad send info: {grad_send_info}")
        return grad_send_info

    def _recv_tensor(self, info, recv_reqs, chunk=None, is_grad=False):
        tag = compute_tag(
            is_grad,
            source_stage_id=info.source,
            receiver_stage_id=self.stage_index,
            chunk=chunk,
            chunks=self.chunks,
            num_stages=self.nstages,
        )
        peer_rank = self.stage_index_to_group_rank[info.source]
        key_type = "grad" if is_grad else "activation"
        key = (key_type, info.source, self.stage_index, chunk, info.input_name)
        logging.debug(
            f"[{self.stage_index}][{self.name}] "
            f"Receiving tensor '{info.input_name}' from Stage {info.source}: "
            f"{info.buffer.size()}"
            f"chunk: {chunk}, is_grad: {is_grad}, peer_rank: {peer_rank}, tag: {tag}"
        )
        if peer_rank == self.group_rank:
            read_done_condition = threading.Condition()
            self.read_done_flags[chunk][key] = False
            work = ReadWork(
                read_done_condition, key, self.read_done_flags[chunk], self.cache_locks[chunk], self.caches[chunk]
            )

            # Define a function to perform the copy operation
            def copy_from_cache():
                logging.debug(f"[{self.stage_index}][{self.name}] " f"read tensor from Cache from Stage {info.source}")
                with self.wait_conditions[chunk]:
                    # Wait for the tensor to be written by the sender
                    while key not in self.caches[chunk]:
                        self.wait_conditions[chunk].wait()

                with self.cache_locks[chunk]:
                    # Copy the tensor to the existing buffer
                    tensor, _ = self.caches[chunk][key]
                    info.buffer.copy_(tensor)
                    self.read_done_flags[chunk][key] = True

                with read_done_condition:
                    read_done_condition.notify_all()

            # Perform the copy operation after waiting for the tensor
            # TODO blocking impl, maybe async?
            # copy_from_cache()
            threading.Thread(target=copy_from_cache).start()
        else:
            # Use async to parallelize recv of tensors
            work = dist.irecv(
                info.buffer,
                peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank),
                group=self.group,
                tag=tag,
            )
        recv_reqs.append(work)
        return info.buffer

    def recv_tensor_fn(self, reqs, chunk=None, is_grad=False):
        return lambda info: self._recv_tensor(info, reqs, chunk, is_grad)

    # Hack here to sieve the correct inputs
    def split_inputs(self, args, kwargs):
        complete_kwargs = dict((key, value) for key, value in zip(self.forward_keys.keys(), args))
        complete_kwargs.update(kwargs)

        submod_args = []
        submod_kwargs = {}

        for name in self.submod_forward_keys:
            if name in complete_kwargs:
                # If the argument is a positional argument, add it to args
                if self.forward_keys[name].default is inspect.Parameter.empty:
                    submod_args.append(complete_kwargs[name])
                else:
                    # If the argument is a keyword argument, add it to kwargs
                    submod_kwargs[name] = complete_kwargs[name]

        self.args_split = None
        self.kwargs_split = None

        if submod_args or submod_kwargs:
            self.args_split, self.kwargs_split = split_args_kwargs_into_chunks(
                submod_args,
                submod_kwargs,
                self.chunks,
                self.args_chunk_spec,
                self.kwargs_chunk_spec,
            )

        self.args_split = self.args_split if len(submod_args) != 0 else None
        self.kwargs_split = self.kwargs_split if len(submod_kwargs) != 0 else None

    def _recv_and_fill_inputs(
        self,
        chunk: int,
    ):
        # Receive requests of a chunk
        recv_reqs: List[Union[dist.Work, ReadWork]] = []

        act_recv = self.recv_tensor_fn(recv_reqs, chunk, is_grad=False)

        if self.args_split:
            chunk_args = self.args_split[chunk]
            chunk_args_list = list(chunk_args)

        def recv_args(info):
            if isinstance(info, RecvInfo):
                return act_recv(info)
            else:
                return chunk_args_list.pop(0)

        try:
            composite_args = pippy.fx.node.map_aggregate(
                self.args_recv_info[chunk],
                recv_args,
            )
        except Exception as e:
            ex_msg = f"""
            Stage {self.stage_index} fails to recv and fill args.
            args_split: {self.args_split}.
            """
            raise RuntimeError(ex_msg) from e

        if self.kwargs_split:
            chunk_kwargs = self.kwargs_split[chunk]

        def recv_kwargs(info):
            if isinstance(info, RecvInfo):
                return act_recv(info)
            else:
                k = next(iter(chunk_kwargs))
                return chunk_kwargs.pop(k)

        composite_kwargs = pippy.fx.node.map_aggregate(
            self.kwargs_recv_info[chunk],
            recv_kwargs,
        )

        # Wait for all recvs to finish
        for work in recv_reqs:
            work.wait()

        return composite_args, composite_kwargs

    def _send_activations(self, output_tuple, chunk=None) -> List[dist.Work]:
        # Send requests of a chunk
        send_reqs: List[Union[dist.Work, SendWork]] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            ref_count = sum(self.stage_index_to_group_rank[dst] == self.group_rank for dst in dst_stages)
            for dst in dst_stages:
                if dst is None:
                    continue
                tag = compute_tag(
                    False,
                    source_stage_id=self.stage_index,
                    receiver_stage_id=dst,
                    chunk=chunk,
                    chunks=self.chunks,
                    num_stages=self.nstages,
                )
                peer_rank = self.stage_index_to_group_rank[dst]
                logging.info(
                    f"[{self.stage_index}][{self.name}] "
                    f"Sending activation to Stage {dst}: {out.size()} "
                    f"chunk: {chunk}, peer_rank: {peer_rank}, tag: {tag}"
                )
                if peer_rank == self.group_rank:
                    key = ("activation", self.stage_index, dst, chunk, self.name)
                    write_done_condition = threading.Condition()
                    self.write_done_flags[chunk][key] = False
                    work = SendWork(write_done_condition, key, self.write_done_flags[chunk])

                    def write_activation():
                        logging.debug(
                            f"[{self.stage_index}][{self.name}] "
                            f"Writes activation to Cache for Stage {dst}: {out.size()}"
                        )
                        with self.cache_locks[chunk]:
                            # Store the tensor in the cache with the reference count
                            # we clone the out tensor to avoid messing up the torch autograd
                            self.caches[chunk][key] = (out.clone(), ref_count)
                            self.write_done_flags[chunk][key] = True

                        with write_done_condition:
                            write_done_condition.notify_all()

                        with self.wait_conditions[chunk]:
                            self.wait_conditions[chunk].notify_all()

                    # TODO blocking impl, maybe async?
                    # write_activation()
                    threading.Thread(target=write_activation).start()

                else:
                    work = dist.isend(
                        out,
                        peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank),  # TODO
                        group=self.group,
                        tag=tag,
                    )
                send_reqs.append(work)

        return send_reqs

    def _recv_grads(
        self,
        bwd_chunk,
    ):
        # Receive requests of a chunk
        grad_recv_reqs: List[Union[dist.Work, ReadWork]] = []

        recv_grad = self.recv_tensor_fn(grad_recv_reqs, bwd_chunk, is_grad=True)

        # Receive gradients
        raw_grads = pippy.fx.node.map_aggregate(
            self.grad_recv_info[bwd_chunk],
            recv_grad,
        )

        # Accumulate gradients for each output index
        grads = {}
        for out_idx, grad_list in raw_grads.items():
            grads[out_idx] = sum(grad_list)

        # Wait for all recvs to finish
        for work in grad_recv_reqs:
            work.wait()

        logging.debug(
            f"[{self.group_rank}][{self.name}] " f"Received output grads of chunk {bwd_chunk}: {map_debug_info(grads)}"
        )
        return grads

    def _send_grads(
        self,
        grads_input,
        chunk,
    ) -> List[dist.Work]:
        # Send requests of a chunk
        grad_send_reqs: List[Union[dist.Work, SendWork]] = []

        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                tag = compute_tag(
                    True,
                    source_stage_id=self.stage_index,
                    receiver_stage_id=grad_recv_stage,
                    chunk=chunk,
                    chunks=self.chunks,
                    num_stages=self.nstages,
                )
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                key = ("grad", self.stage_index, grad_recv_stage, chunk, self.name)
                logging.debug(
                    f"[{self.stage_index}][{self.name}] "
                    f"Sending grad to Stage {grad_recv_stage}: {grad.size()} "
                    f"chunk: {chunk}, peer_rank: {peer_rank}, tag: {tag}"
                )
                if peer_rank == self.group_rank:  # Same-rank transfer
                    write_done_condition = threading.Condition()
                    self.write_done_flags[chunk][key] = False
                    work = SendWork(write_done_condition, key, self.write_done_flags[chunk])

                    def write_grad():
                        logging.debug(
                            f"[{self.stage_index}][{self.name}] "
                            f"Writes gradient to Cache for Stage {grad_recv_stage}: {grad.size()}"
                        )
                        with self.cache_locks[chunk]:
                            # Store the tensor in the cache
                            self.caches[chunk][key] = (grad.clone(), 1)
                            self.write_done_flags[chunk][key] = True

                        with write_done_condition:
                            write_done_condition.notify_all()

                        with self.wait_conditions[chunk]:
                            self.wait_conditions[chunk].notify_all()

                    # TODO blocking impl, maybe async?
                    # write_grad()
                    threading.Thread(target=write_grad).start()
                else:
                    work = dist.isend(
                        grad,
                        peer_rank if self.group is None else dist.get_global_rank(self.group, peer_rank),  # TODO
                        group=self.group,
                        tag=tag,
                    )
                grad_send_reqs.append(work)
            else:
                assert grad is None and grad_recv_stage is None

        return grad_send_reqs

    def forward_maybe_with_nosync(self, *args, **kwargs):
        # If submod is wrapped with DDP, we use the `no_sync` context manager to
        # avoid gradient all-reduce per microbatch
        if isinstance(self.submod, DistributedDataParallel):
            with self.submod.no_sync():  # type: ignore[operator]
                out_val = self.submod(*args, **kwargs)
        else:
            out_val = self.submod(*args, **kwargs)
        return out_val

    def backward_maybe_with_nosync(self, bwd_kwargs: Dict, is_last_chunk: bool):
        if isinstance(self.submod, DistributedDataParallel):
            if is_last_chunk:
                # HACK: reaching into DDP implementation details here. Is there a better way?
                self.submod.reducer.prepare_for_backward(  # type: ignore[union-attr, operator]
                    list(
                        torch.nn.parallel.distributed._find_tensors(  # type: ignore[attr-defined]
                            bwd_kwargs["stage_output"]
                        )
                    )
                )
                grads_input, _ = stage_backward(**bwd_kwargs)
            else:
                with self.submod.no_sync():  # type: ignore[operator]
                    grads_input, _ = stage_backward(**bwd_kwargs)
        else:
            # Non-DDP submodule, regular backward
            # if torch.distributed.get_rank() == 1:
            grads_input, _ = stage_backward(**bwd_kwargs)
        return grads_input

    def forward_one_chunk(
        self,
        chunk: int,
    ):
        composite_args, composite_kwargs = self._recv_and_fill_inputs(
            chunk,
        )

        # Compute forward
        try:
            if self.checkpoint:
                with torch.no_grad():
                    output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)
            else:
                with torch.enable_grad():
                    output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)

        except Exception as e:
            exc_msg = f"""
            Rank {self.group_rank} failed to run forward stage: {self.name}
            args: {map_debug_info(composite_args)}
            kwargs: {map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # Unify output form to tuple for easy correspondance with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        self.output_chunks.append(output)
        send_reqs = self._send_activations(output_tuple, chunk=chunk)
        self.all_act_send_reqs += send_reqs

        if self.checkpoint:
            self.fwd_cache[chunk] = (composite_args, composite_kwargs)
        else:
            # Save activations and inputs for backward
            flat_args = flatten_args(composite_args)
            flat_kwargs = flatten_args(composite_kwargs)
            flatten_input_tensors = flat_args + flat_kwargs

            self.fwd_cache[chunk] = (
                output_tuple,  # stage_output
                flatten_input_tensors,  # input_values
            )

    def backward_one_chunk(
        self,
        bwd_chunk: int,
    ):
        if not self.pipe.has_loss_and_backwards:
            return None

        grads = self._recv_grads(bwd_chunk)

        # Pack args for `stage_backward``
        bwd_kwargs = dict(self.bwd_node.kwargs)
        if self.checkpoint:
            composite_args, composite_kwargs = self.fwd_cache[bwd_chunk]
            with torch.enable_grad():
                output = self.forward_maybe_with_nosync(*composite_args, **composite_kwargs)
            output_tuple = output if type(output) is tuple else (output,)
            # Save activations and inputs for backward
            flat_args = flatten_args(composite_args)
            flat_kwargs = flatten_args(composite_kwargs)
            flatten_input_tensors = flat_args + flat_kwargs
            (
                bwd_kwargs["stage_output"],
                bwd_kwargs["input_values"],
            ) = (output_tuple, flatten_input_tensors)
        else:
            (
                bwd_kwargs["stage_output"],
                bwd_kwargs["input_values"],
            ) = self.fwd_cache.pop(bwd_chunk)

        # Fill actual gradients received for outputs
        # If nothing received, as in the case of last stage, then we
        # would use the default `output_grads` prepared in the IR phase,
        # i.e. from `bwd_node.kwargs`. For example, it may look like
        # this if there are two outputs: ('None', 'None')
        if len(grads) > 0:
            diff_count = 0
            output_grads = list(bwd_kwargs["output_grads"])
            for idx, stage_out in enumerate(bwd_kwargs["stage_output"]):
                if stage_out.requires_grad:
                    output_grads[idx] = grads[diff_count]
                    diff_count += 1
                else:
                    output_grads[idx] = None
            bwd_kwargs["output_grads"] = tuple(output_grads)

        # `stage_backward` node does not have `args`, only `kwargs`
        grads_input = self.backward_maybe_with_nosync(
            bwd_kwargs,
            bwd_chunk == self.chunks - 1,
        )

        grad_send_reqs = self._send_grads(grads_input, chunk=bwd_chunk)
        self.all_grad_send_reqs += grad_send_reqs

    def clear_runtime_states(self):
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Activation send requests of all chunk
        self.all_act_send_reqs.clear()
        # Grad send requests of all chunk
        self.all_grad_send_reqs.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()
        for chunk in range(self.chunks):
            self.read_done_flags[chunk].clear()

        for chunk in range(self.chunks):
            self.write_done_flags[chunk].clear()

    def merge_output_chunks(self):
        return merge_chunks(
            self.output_chunks,
            self.output_chunk_spec,
        )

    def forward(self, *args, **kwargs):
        # map microbatch ID to list of forward tensor args
        # Clean per iteration
        self.clear_runtime_states()

        # Split inputs into chunks
        self.split_inputs(args, kwargs)

        # Forward pass of all chunks
        for chunk in range(self.chunks):
            self.forward_one_chunk(chunk)

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_send_reqs:
            work.wait()

        # Backward starts here

        for bwd_chunk in range(self.chunks):
            self.backward_one_chunk(bwd_chunk)

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_grad_send_reqs:
            work.wait()

        # Last rank return merged results per original format
        if self.is_last():
            return self.merge_output_chunks()
        else:
            return None


class PipelineStage1F1B(PipelineStage):
    def __init__(
        self,
        pipe: Pipe,
        stage_index: int,
        nstages: int,
        chunks: int,
        device: torch.device,
        checkpoint: bool = False,
        group: dist.ProcessGroup = None,
        args_chunk_spec=None,
        kwargs_chunk_spec=None,
        output_chunk_spec=None,
        forward_keys=None,
    ):
        super().__init__(
            pipe,
            stage_index,
            nstages,
            chunks,
            device,
            checkpoint=checkpoint,
            group=group,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec,
            forward_keys=forward_keys,
        )

    def forward(self, *args, **kwargs):
        # Clean per iteration
        self.clear_runtime_states()

        # Split inputs into chunks
        self.split_inputs(args, kwargs)

        warmup_chunks = cooldown_chunks = self.nstages

        # Warm-up phase: forward number of chunks equal to pipeline depth.
        for chunk in range(warmup_chunks):
            self.forward_one_chunk(chunk)

        # 1F1B phase
        for bwd_chunk in range(0, self.chunks - cooldown_chunks):
            # Schedule backward for one warmed up chunk
            self.backward_one_chunk(bwd_chunk)

            # Schedule forward for one new chunk
            fwd_chunk = bwd_chunk + warmup_chunks
            self.forward_one_chunk(fwd_chunk)

        # Cool-down phase: backward for the rest of the chunks
        for bwd_chunk in range(self.chunks - cooldown_chunks, self.chunks):
            self.backward_one_chunk(bwd_chunk)

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for work in self.all_send_reqs:
            work.wait()

        for work in self.all_grad_send_reqs:
            work.wait()

        # Last rank return merged results per original format
        if self.is_last():
            return self.merge_output_chunks()
        else:
            return None
