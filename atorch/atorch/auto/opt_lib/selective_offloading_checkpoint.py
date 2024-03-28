"""This file is for selecting activation recomputing.
We offload activation to cpu, in recomputing stage, reload to gpu.
refers to https://github.com/pytorch/pytorch/issues/70135#issuecomment-1542439983
docs: https://yuque.antfin-inc.com/ai-infra/atorch-design/os05u91bdresusku
"""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import DefaultDict, Deque, List, Optional, Set

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

import atorch

try:
    # _FreeEventQueue is available for pytorch version >=2.0.0
    from torch.distributed.fsdp._limiter_utils import _FreeEventQueue

    class FreeEventQueue(_FreeEventQueue):  # type: ignore
        def deque_event_and_synchronize(self):
            event = self.dequeue_if_needed()
            if event:
                event.synchronize()

        def enque_event(self):
            free_event = torch.cuda.Event()
            free_event.record()
            self.enqueue(free_event)

except (ImportError, ModuleNotFoundError):

    class FreeEventQueue:  # type: ignore
        """https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_limiter_utils.py
        Maybe we do not need limit copy.
        """

        def __init__(self):
            self._queue = deque()
            self._max_num_inflight_copy = 2

        def enqueue(self, free_event):
            self._queue.append(free_event)

        def dequeue_if_needed(self):
            if len(self._queue) >= self._max_num_inflight_copy:
                return self._dequeue()
            return None

        def _dequeue(self):
            if self._queue:
                event = self._queue.popleft()
                return event
            return None

        def deque_event_and_synchronize(self):
            event = self.dequeue_if_needed()
            if event:
                event.synchronize()

        def enque_event(self):
            free_event = torch.cuda.Event()
            free_event.record()
            self.enqueue(free_event)


@dataclass
class OffloadOpManagerArgs:
    """Args for `OffloadOpManager`
    name: tell torch which op to offload to cpu, we test matmul, named 'mm.default'
    index_to_offload: We needs to balance `compute bandwidth` and `upstream bandwidth in pcie`, so we can not
        offload all activation to cpu, so this param is for which op instance needs to be offload.
        eg. We have 10 matmul, we want to offload first, third matmul op, it should pass [1,3].
        In a 8 gpus node, 2 gpus are sharing upstream bandwidth, so we should let those gpus offload different
        activations to avoid pcie race.
    """

    name: str
    index_to_offload: List[int]
    index_to_hold: Optional[List[int]] = field(default_factory=list)  # type: ignore
    index_to_hold_layer: Optional[int] = -1


class OffloadOpManager:
    def __init__(self, config: OffloadOpManagerArgs):
        """
        name: See `OffloadOpManagerArgs`
        offload_index: recording which op instance is processing.
        reload_index: recording which op instance is processing.
        cpu_storage: fifo cache, cpu memory to store offloaded activation.
        gpu_storage: fifo cache, cuda memory buffer for this offload, we want to avoid uesless offload of
            last wrapped layer.
        shaped_cpu_cache: fifo cache, cached pin_memory buffer for this offload.

        .. warning::

            In some layer, the last op instance may not needs to recompute, this occurs when this op is used for
            computing grad directlly. eg. The matmul is last op in forward or activation of op is first used.
            At this point, `cpu_storage` cache offload via `index_to_offload`, it don't know which op should be
            recomputed in backward, this will lead useless allocation and wrong offload.
        """
        self.name: str = config.name
        self.offload_index: int = 0
        self.reload_index: int = 0
        self.index_to_offload: Set[int] = set(config.index_to_offload)
        assert config.index_to_hold is not None  # for mypy
        self.index_to_hold: Set[int] = set(config.index_to_hold)
        self.index_to_hold_layer = config.index_to_hold_layer
        self.cpu_storage: Deque[torch.Tensor] = deque()
        self.gpu_storage: Deque[torch.Tensor] = deque()
        self.shaped_cpu_cache: DefaultDict[torch.Size, Deque[torch.Tensor]] = defaultdict(deque)

    def need_hold_on_gpu_by_layer(self, layer_index, index):
        return layer_index < self.index_to_hold_layer and index in self.index_to_hold

    def reset(self):
        """Reseting all index to 0, this should called in each iteration."""
        self.offload_index = self.reload_index = 0

    def offload(self, x, offload_event_queue, current_stream, copy_stream, layer_index, need_hold_on_gpu=False):
        """Offload gpu to cpu. We use `offload_index` and `index_to_offload` to select which to offload.
        If `need_hold_on_gpu` is True, we use `gpu_storage` to cache tensor and pop in backward.
        """
        self.offload_index += 1
        if self.offload_index not in self.index_to_offload and self.offload_index not in self.index_to_hold:
            return

        if need_hold_on_gpu or self.need_hold_on_gpu_by_layer(layer_index, self.offload_index):
            self.gpu_storage.append(x)
            return

        def _detach_to_cpu(x):
            if isinstance(x, torch.Tensor) and x.is_cuda:
                offload_event_queue.deque_event_and_synchronize()
                tensor = x.detach()
                copy_stream.wait_stream(current_stream)
                s = tensor.shape
                if not self.shaped_cpu_cache[s]:
                    packed = torch.empty_like(tensor, device="cpu", pin_memory=True)
                else:
                    packed = self.shaped_cpu_cache[s].popleft()
                with torch.cuda.stream(copy_stream):
                    packed.copy_(tensor, non_blocking=True)
                tensor.record_stream(copy_stream)
                offload_event_queue.enque_event()
                return packed

        out_detached_cpu = tree_map(_detach_to_cpu, x)
        self.cpu_storage.append(out_detached_cpu)

    def reload(self, x_gen, reload_event_queue, current_stream, copy_stream, layer_index, need_hold_on_gpu=False):
        """Same as `offload`."""
        self.reload_index += 1
        if self.reload_index not in self.index_to_offload and self.reload_index not in self.index_to_hold:
            return x_gen()

        if need_hold_on_gpu or self.need_hold_on_gpu_by_layer(layer_index, self.reload_index):
            return self.gpu_storage.popleft()

        def _to_cuda(x):
            if isinstance(x, torch.Tensor) and x.device.type == "cpu":
                s = x.shape
                reload_event_queue.deque_event_and_synchronize()
                self.shaped_cpu_cache[s].append(x)
                with torch.cuda.stream(copy_stream):
                    unpacked = x.to(f"cuda:{atorch.local_rank()}", non_blocking=True)
                current_stream.wait_stream(copy_stream)
                unpacked.record_stream(current_stream)
                reload_event_queue.enque_event()
                return unpacked
            return x

        out = tree_map(_to_cuda, self.cpu_storage.popleft())
        return out


def get_selective_offloading_checkpoint_modes(offload_args: List[OffloadOpManagerArgs], num_of_layers: int):
    """Generate arg `context_fn` in `torch.utils.checkpoint`.
    Maybe use some algo to generate `offload_args` automatically.
    """
    copy_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    pack_event_queue = FreeEventQueue()
    unpack_event_queue = FreeEventQueue()
    current_offload_index = 0

    class CachingMode(TorchDispatchMode):
        """Doing dispatch in forward pass, inspect kernel and use `OffloadOpManager` to offload tensor."""

        def __init__(self, offloaders, num_of_layers, index, _dispatch_key=None):
            self.offloaders = offloaders
            self.need_hold_on_gpu = index == num_of_layers
            self.index = index
            super().__init__(_dispatch_key)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            out = func(*args, **kwargs)
            func_name = func.__name__
            if func_name not in self.offloaders:
                return out
            offloader = self.offloaders[func_name]
            offloader.offload(out, pack_event_queue, current_stream, copy_stream, self.index, self.need_hold_on_gpu)
            return out

    class CachedMode(TorchDispatchMode):
        """Doing dispatch in backward pass, inspect kernel and use `OffloadOpManager` to reload tensor."""

        def __init__(self, offloaders, num_of_layers, index, _dispatch_key=None):
            self.offloaders = offloaders
            self.need_hold_on_gpu = index == num_of_layers
            self.index = index
            super().__init__(_dispatch_key)

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = {} if kwargs is None else kwargs
            func_name = func.__name__
            if func_name not in self.offloaders:
                return func(*args, **kwargs)
            offloader = self.offloaders[func_name]

            def x_gen():
                return func(*args, **kwargs)

            out = offloader.reload(
                x_gen, unpack_event_queue, current_stream, copy_stream, self.index, self.need_hold_on_gpu
            )
            return out

    def gen(offload_args: List[OffloadOpManagerArgs], num_of_layers: int):
        """Generating instance of `CachingMode` and ``CachedMode`."""
        nonlocal current_offload_index
        current_offload_index += 1
        offloaders = {arg.name: OffloadOpManager(arg) for arg in offload_args}
        caching_mode = CachingMode(offloaders, num_of_layers, current_offload_index)
        cached_mode = CachedMode(offloaders, num_of_layers, current_offload_index)
        return caching_mode, cached_mode

    offload_objs = [gen(offload_args, num_of_layers) for _ in range(num_of_layers)]

    def inner():
        """Real selective policy."""
        nonlocal current_offload_index
        if current_offload_index == num_of_layers:
            current_offload_index = 0
        caching_mode, cached_mode = offload_objs[current_offload_index]
        for offload in caching_mode.offloaders.values():
            offload.reset()
        current_offload_index += 1
        return caching_mode, cached_mode

    return inner
