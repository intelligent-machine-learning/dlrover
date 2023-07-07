# Copyright 2023 The DLRover Authors. All rights reserved.
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

import contextlib
import os
import socket
from contextlib import contextmanager
from typing import Any, Dict

import torch
import torch.distributed as dist

from dlrover.python.common.log import default_logger as logger


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]


def get_rank():
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    return rank


class GradientState(object):
    """
    Singleton class that has information related to gradient
    synchronization for gradient accumulation.

    **Available attributes:**
        - *sync_gradients** (`bool`) -- Whether gradients are currently
            being synchronized across all processes.
        - *num_backward_steps** (`int`) -- The number of step to perform
            backward to compute gradients.
        - *num_steps** (`int`) -- The number of step to sync gradients and
            update model parameters by gradients.
    """

    _shared_state: Dict[str, Any] = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self.sync_gradients = True
            self.num_backward_steps = 0
            self.num_steps = 0

    def check_sync_gradient(self, gradient_accumulation_steps):
        """Check whether to synchronize gradients accross all processes."""
        if self.num_backward_steps % gradient_accumulation_steps == 0:
            self.sync_gradients = True
        else:
            self.sync_gradients = False

    @property
    def initialized(self) -> bool:
        "Returns whether the `GradientState` has been initialized"
        return GradientState._shared_state != {}


class _ElasticOptimizer(torch.optim.Optimizer):
    """
    Internal wrapper around a torch optimizer.
    Perform `step` and `zero_grad` if gradients should be synchronized.
    Args:
        optimizer (`torch.optim.optimizer.Optimizer`):
            The optimizer to wrap.
    """

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.gradient_state = GradientState()

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, param_groups):
        self.optimizer.param_groups = param_groups

    @property
    def defaults(self):
        return self.optimizer.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optimizer.defaults = defaults

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self, set_to_none=None):
        if self.gradient_state.sync_gradients:
            self.optimizer.zero_grad(set_to_none)

    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            self.optimizer.step(closure)

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class _ElasticLRScheduler(object):
    """A wrapper around a learning rate scheduler that will only
    step after the gradients are sychronizing across all processes and
    the optimizer steps."""

    def __init__(
        self, scheduler: torch.optim.lr_scheduler._LRScheduler
    ) -> None:
        self.scheduler = scheduler
        self.gradient_state = GradientState()

    def step(self, *args, **kwargs):
        self.scheduler._step_count = self.gradient_state.num_steps
        self.scheduler.step(*args, **kwargs)

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    def get_lr(self):
        return self.scheduler.get_lr()

    def print_lr(self, *args, **kwargs):
        return self.scheduler.print_lr(*args, **kwargs)


class ElasticTrainer(object):
    """Creates an instance of an elastic trainer for elastic distributed
    training on multi-nodes. The elastic trainer will do:
    - set the number of step to accumlate gradients to keep the global
    batch size fixed.
    - do checkpoint before the worker group changes.

    Args:
        model (`torch.nn.Module`): PyTorch Module.

    **Available attributes:**
        - **step** -- the number of local step on the process.

        - **gradient_accumulation_steps** (`int`): The number of
            steps that should pass before gradients are accumulated.
            It will change with the number of workers. For example,
            if the expected max number of worker is 8, it is 4 if the current
            number of worker is 2 and it is 2 if the current number of
            worker is 4. The elastic trainer can keep the global batch
            size fixed by adjusting the gradient_accumulation_steps.
    """

    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.gradient_state = GradientState()
        self.gradient_accumulation_steps = 1

    def prepare(self, optimizer, lr_scheduler=None):
        """
        Prepare optimizer and learning rate scheduler in elastic training.
        """
        self._set_gradient_accumulation_steps()
        optimizer = _ElasticOptimizer(optimizer)
        if lr_scheduler:
            lr_scheduler = _ElasticLRScheduler(lr_scheduler)
            return optimizer, lr_scheduler
        else:
            return optimizer

    @contextmanager
    def step(self, fix_total_batch_size=True):
        """
        A context manager that will lightly wrap around and to keep
        the global batch size fixed when the number of worker changes.

        Args:
            fix_total_batch_size (`bool`): Whether to keep the total
            batch size fixed when the optimizer steps. If True,
            the context manager will perform gradient accumulation and
            synchronize gradients when the accumulated batch size if
            equal to the global batch size.

        Example:
        ```python
        >>> from dlrover.trainer.torch.elastic import ElasticTrainer
        >>> elastic_trainer = ElasticTrainer(model)
        >>> optimizer, scheduler = elastic_trainer.prepare(
                optimizer, scheduler
            )
        >>> for input, output in dataloader:
        ...     with elastic_trainer.step():
        ...         outputs = model(input)
        ...         loss = loss_func(outputs)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()
        ...         optimizer.zero_grad()
        ```
        """
        self._before_step(fix_total_batch_size)
        context = contextlib.nullcontext
        if not self.gradient_state.sync_gradients:
            context = getattr(self.model, "no_sync", context)

        with context():
            yield
            self._after_step()

    def reset(self):
        self.gradient_state.num_steps = 0

    @property
    def num_steps(self):
        return self.gradient_state.num_steps

    def _before_step(self, fix_total_batch_size):
        """Sets the right `sync_gradients` and either resets
        or increases `self.step`"""
        self.gradient_state.num_backward_steps += 1
        if not fix_total_batch_size:
            self.gradient_state.sync_gradients = True
        else:
            self.gradient_state.check_sync_gradient(
                self.gradient_accumulation_steps
            )

    def _after_step(self):
        if self.gradient_state.sync_gradients:
            self.gradient_state.num_steps += 1

    def _set_gradient_accumulation_steps(self):
        max_worker_num = int(os.getenv("WORKER_NUM", 0))
        if max_worker_num == 0:
            self.gradient_accumulation_steps = 1

        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        max_worker_num *= local_size
        cur_world_size, rank = 1, 0
        if dist.is_initialized():
            cur_world_size = dist.get_world_size()
            rank = dist.get_rank()
        self.gradient_accumulation_steps = int(max_worker_num / cur_world_size)
        remainder = max_worker_num % cur_world_size
        if rank < remainder:
            self.gradient_accumulation_steps += 1
        logger.info(
            "Rank = %s, World size = %s, Gradient accumulation steps = %s",
            rank,
            cur_world_size,
            self.gradient_accumulation_steps,
        )
