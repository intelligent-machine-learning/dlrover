# Copyright (c) Meta Platforms, Inc. and affiliates
import threading
from typing import Any, Dict, List, Tuple

import torch

from .PipelineStage import PipelineStage

_LRScheduler = None
try:
    from torch.optim.lr_scheduler import _LRScheduler  # type: ignore
except ImportError:
    _LRScheduler = object


class StageInterleaver(torch.nn.Module):
    def __init__(
        self,
        stages: List[PipelineStage],
    ):
        super().__init__()
        self.stages = stages
        self.num_total_stages = stages[0].nstages
        self.chunks = stages[0].chunks
        self.caches: List[Dict[int, Tuple[Any]]] = [dict() for _ in range(self.chunks)]
        self.cache_locks = [threading.Lock() for _ in range(self.chunks)]
        self.wait_conditions = [threading.Condition() for _ in range(self.chunks)]
        for stage in stages:
            stage.set_cache(self.caches, self.cache_locks, self.wait_conditions)

    def has_first(self):
        for stage in self.stages:
            if stage.is_first():
                return True
        return False

    def has_last(self):
        for stage in self.stages:
            if stage.is_last():
                return True
        return False

    def forward(self, *args, **kwargs):
        # Clean per iteration
        for stage in self.stages:
            stage.clear_runtime_states()

        # Clearing the cache at the beginning of each forward computation
        for lock, cache in zip(self.cache_locks, self.caches):
            with lock:
                cache.clear()

        # Split inputs into chunks
        for stage in self.stages:
            stage.split_inputs(args, kwargs)

        # TODO: verify
        # Equal to half of pipeline depth.
        warmup_chunks = cooldown_chunks = self.num_total_stages // 2

        # Warm-up phase: forward some chunks for each stage.
        for stage in self.stages:
            for chunk in range(warmup_chunks):
                stage.forward_one_chunk(chunk)

        # 1F1B phase
        for stage in self.stages:
            for bwd_chunk in range(0, self.chunks - cooldown_chunks):
                # Schedule backward for one warmed up chunk
                stage.backward_one_chunk(bwd_chunk)

                # Schedule forward for one new chunk
                fwd_chunk = bwd_chunk + warmup_chunks
                stage.forward_one_chunk(fwd_chunk)

        # Cool-down phase: backward for the rest of the chunks
        for stage in self.stages:
            for bwd_chunk in range(self.chunks - cooldown_chunks, self.chunks):
                stage.backward_one_chunk(bwd_chunk)

        # Wait for all sends to finish
        # TODO: okay to delay the sync till completion of all chunks?
        for stage in self.stages:
            for work in stage.all_act_send_reqs:
                work.wait()

        for stage in self.stages:
            for work in stage.all_grad_send_reqs:
                work.wait()

        # Last rank return merged results per original format
        last_stage = None
        for stage in self.stages:
            # TODO: extend qualification for returning outputs
            if stage.is_last():
                last_stage = stage
                break

        if last_stage is None:
            return None
        else:
            return last_stage.merge_output_chunks()


class InterleaverOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self, *args, **kwargs):
        for optim in self.optimizers:
            optim.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        for optim in self.optimizers:
            optim.step(*args, **kwargs)


class InterleaverLRScheduler(_LRScheduler):  # type: ignore
    def __init__(self, lr_schedulers):
        self.lr_schedulers = lr_schedulers

    def step(self, *args, **kwargs):
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
