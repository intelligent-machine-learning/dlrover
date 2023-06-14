# coding=utf-8
# from torch.distributed.utils import (
# _verify_param_shape_across_processes,
# _sync_module_states,
# )
import copy
import logging
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _DDPSink, _find_tensors, _tree_flatten_with_rref, _tree_unflatten_with_rref

from atorch.common.log_utils import default_logger as logger
from atorch.modules.moe.moe_layer import get_experts_ddp_process_group

# torch 2.0
try:
    from torch.distributed.utils import _sync_module_states
except (ImportError, ModuleNotFoundError):
    _sync_module_states = None


class MoEMixtureDistributedDataParallel(DistributedDataParallel):
    """
    when there have moe,they need different DDP mode
    copy DDP doc:
    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    example:
    we have 8 ranks.
    [0,1,2,3,4,5,6,7] there dense params are one process_group, they all_reduce gradients
    """

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    ):
        moe_params_names, dense_params_names = self.get_moe_or_dense_params(module)

        # sync dense_params between rank
        module._ddp_params_and_buffers_to_ignore = moe_params_names
        self.dense_params_names = set(dense_params_names)
        super().__init__(
            module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            process_group=process_group,
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        # we need other 1. device_ids 2. process_group
        self.experts_process_group = get_experts_ddp_process_group()
        assert self.experts_process_group is not None, "Must `set_experts_process_group` before using MoeDDP"

        # save origin ddp reducer and logger
        origin_ddp_reducer = self.reducer
        origin_ddp_logger = self.logger
        origin_ddp_rank = dist.get_rank(self.process_group)
        origin_ddp_process_group = self.process_group
        with self.moe_context(self.dense_params_names, self.experts_process_group):
            # temp modify
            # 1. self.parameters_to_ignore,
            # 2. process_groupchange to dense params, to get moe params
            moe_parameters, moe_expect_sparse_gradient = self._build_params_for_reducer()
            moe_param_numel = sum(
                sum(t.numel() for t in p) if isinstance(p, (list, tuple)) else p.numel() for p in moe_parameters
            )
            if moe_param_numel:
                # boardcast experts params
                logger.info("moeDDP[%d]: _verify_model_across_ranks numel:%d", origin_ddp_rank, moe_param_numel)
                if hasattr(dist, "_verify_params_across_processes"):  # torch 2.0
                    dist._verify_params_across_processes(origin_ddp_process_group, moe_parameters)
                else:  # torch 1.13
                    dist._verify_model_across_ranks(origin_ddp_process_group, moe_parameters)
                logger.info("moeDDP[%d]: _verify_model_across_ranks finish", origin_ddp_rank)
                # using moe_process_group to sync; 0 means: moe_rank=0
                if _sync_module_states is not None:
                    _sync_module_states(
                        module=self.module,
                        process_group=self.process_group,
                        broadcast_bucket_size=self.broadcast_bucket_size,
                        src=0,
                        params_and_buffers_to_ignore=self.parameters_to_ignore,
                    )
                else:  # torch 1.10
                    self._sync_params_and_buffers(authoritative_rank=0)
                logger.info("moeDDP[%d]: _sync_params_and_buffers finish", origin_ddp_rank)

                # In debug mode, build a mapping of parameter index -> parameter.
                moe_param_to_name_mapping = self._build_debug_param_to_name_mapping(moe_parameters)

                # Builds moe reducer.
                self._ddp_init_helper(
                    moe_parameters,
                    moe_expect_sparse_gradient,
                    moe_param_to_name_mapping,
                    static_graph=static_graph,  # 1.10have no static_graph?
                )
                logger.info("moeDDP[%d]: _ddp_init_helper finish", origin_ddp_rank)
            else:
                logger.warning("moeDDP[%d] no moe params,skip ddp init", origin_ddp_rank)
            # restore
            self.moe_reducer = self.reducer
            self.moe_logger = self.logger

        self.reducer = origin_ddp_reducer
        self.logger = origin_ddp_logger
        # TODO:
        # 1. do something _in_backward_optimizers ,
        # 2. do something sync buffer

    @contextmanager
    def moe_context(self, dense_param_name, moe_process_group):
        old_parameters_to_ignore = self.parameters_to_ignore
        origin_process_group = self.process_group
        self.parameters_to_ignore = dense_param_name
        self.process_group = moe_process_group
        try:
            yield
        finally:
            self.parameters_to_ignore = old_parameters_to_ignore
            self.process_group = origin_process_group

    @staticmethod
    def get_moe_or_dense_params(module):
        moe_params_names = []
        dense_params_names = []
        experts_numel = 0
        total_numel = 0
        for name, tensor in module.named_parameters():
            if "mlp.experts" in name:
                moe_params_names.append(name)
                experts_numel += tensor.numel()
            else:
                dense_params_names.append(name)
            total_numel += tensor.numel()
        return moe_params_names, dense_params_names

    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs["process_group"]
        del attrs["reducer"]
        del attrs["logger"]
        del attrs["experts_process_group"]
        del attrs["moe_reducer"]
        del attrs["moe_logger"]

        if self._use_replicated_tensor_module:
            del attrs["_replicated_tensor_module"]
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        super(MoEMixtureDistributedDataParallel, self).__setstate__(state)
        _, dense_params_names = self.get_moe_or_dense_params(self.module)
        # save origin ddp reducer and logger
        origin_ddp_reducer = self.reducer
        origin_ddp_logger = self.logger
        # temp modify self.parameters_to_ignore,change to dense params, to get moe params
        self.experts_process_group = get_experts_ddp_process_group()
        self.dense_params_names = dense_params_names
        with self.moe_context(self.dense_params_names, self.experts_process_group):

            moe_parameters, moe_expect_sparse_gradient = self._build_params_for_reducer()
            param_to_name_mapping = self._build_debug_param_to_name_mapping(moe_parameters)

            # Builds reducer.
            self._ddp_init_helper(
                moe_parameters,
                moe_expect_sparse_gradient,
                param_to_name_mapping,
                self.static_graph,
            )
        # restore
        self.moe_reducer = self.reducer
        self.moe_logger = self.logger

        self.reducer = origin_ddp_reducer
        self.logger = origin_ddp_logger

    def _set_static_graph(self):
        super()._set_static_graph()
        self.moe_reducer._set_static_graph()
        self.moe_logger._set_static_graph()

    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("MoEMixtureDistributedDataParallel.forward"):
            if torch.is_grad_enabled() and self.require_backward_grad_sync:
                self.logger.set_runtime_stats_and_log()
                self.moe_logger.set_runtime_stats_and_log()
                self.num_iterations += 1
                self.reducer.prepare_for_forward()
                self.moe_reducer.prepare_for_forward()

            # Notify the join context that this process has not joined, if
            # needed
            work = Join.notify_join_context(self)
            if work:
                self.reducer._set_forward_pass_work_handle(work, self._divide_by_initial_world_size)
                self.moe_reducer._set_forward_pass_work_handle(work, self._divide_by_initial_world_size)

            # Calling _rebuild_buckets before forward compuation,
            # It may allocate new buckets before deallocating old buckets
            # inside _rebuild_buckets. To save peak memory usage,
            # call _rebuild_buckets before the peak memory usage increases
            # during forward computation.
            # This should be called only once during whole training period.
            if torch.is_grad_enabled() and (self.reducer._rebuild_buckets() or self.moe_reducer._rebuild_buckets()):
                logging.info("Reducer buckets have been rebuilt in this iteration.")
                self._has_rebuilt_buckets = True

            if self.require_forward_param_sync:
                if hasattr(self, "_sync_buffers"):
                    self._sync_buffers()
                else:
                    self._sync_params()  # buffer sync;

            if self._join_config.enable:
                # Notify joined ranks whether they should sync in backwards pass or not.
                self._check_global_requires_backward_grad_sync(is_joined_rank=False)

            if self.device_ids:
                inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
                output = self.module(*inputs[0], **kwargs[0])
            else:
                output = self.module(*inputs, **kwargs)

            if torch.is_grad_enabled() and self.require_backward_grad_sync:
                self.require_forward_param_sync = True
                # We'll return the output object verbatim since it is a freeform
                # object. We need to find any tensors in this object, though,
                # because we need to figure out which parameters were used during
                # this forward pass, to ensure we short circuit reduction for any
                # unused parameters. Only if `find_unused_parameters` is set.
                if self.find_unused_parameters and not self.static_graph:
                    # Do not need to populate this for static graph.
                    self.reducer.prepare_for_backward(list(_find_tensors(output)))
                    self.moe_reducer.prepare_for_backward(list(_find_tensors(output)))
                else:
                    self.reducer.prepare_for_backward([])
                    self.moe_reducer.prepare_for_backward([])
            else:
                self.require_forward_param_sync = False

        # TODO: DDPSink is currently enabled for unused parameter detection and
        # static graph training for first iteration.
        if (self.find_unused_parameters and not self.static_graph) or (self.static_graph and self.num_iterations == 1):
            state_dict = {
                "static_graph": self.static_graph,
                "num_iterations": self.num_iterations,
            }

            output_tensor_list, treespec, output_is_rref = _tree_flatten_with_rref(output)
            output_placeholders = [None for _ in range(len(output_tensor_list))]
            # Do not touch tensors that have no grad_fn, which can cause issues
            # such as https://github.com/pytorch/pytorch/issues/60733
            for i, output in enumerate(output_tensor_list):
                if torch.is_tensor(output) and output.grad_fn is None:
                    output_placeholders[i] = output

            # When find_unused_parameters=True, makes tensors which require grad
            # run through the DDPSink backward pass. When not all outputs are
            # used in loss, this makes those corresponding tensors receive
            # undefined gradient which the reducer then handles to ensure
            # param.grad field is not touched and we don't error out.
            passthrough_tensor_list = _DDPSink.apply(
                self.reducer,
                state_dict,
                *output_tensor_list,
            )
            _DDPSink.apply(
                self.moe_reducer,
                state_dict,
                *output_tensor_list,
            )
            for i in range(len(output_placeholders)):
                if output_placeholders[i] is None:
                    output_placeholders[i] = passthrough_tensor_list[i]

            # Reconstruct output data structure.
            output = _tree_unflatten_with_rref(output_placeholders, treespec, output_is_rref)
        return output

    def _match_all_reduce_for_bwd_pass(self):
        comm_work = []
        # Schedule comm in the same order as Reducer schedules them, i.e.
        # the order of the buckets. Retrieving the bucket order from the reducer
        # ensures that we keep the same order in join mode, such as when bucket
        # order is rebuilt dynamically.

        # Returns grad_buckets in order, but real tensors are substituted with
        # zero tensors of the same shape.
        grad_buckets = self.reducer._get_zeros_like_grad_buckets()
        moe_grad_buckets = self.moe_reducer._get_zeros_like_grad_buckets()
        for grad_bucket in grad_buckets:
            # Joined processes contribute zero gradient. In the case that
            # divide_by_initial_world_size=True, we divide grads by the static
            # world size, if not, the dividing factor is reduced by the number
            # of joined processes.
            work = self.reducer._run_comm_hook(grad_bucket)
            comm_work.append(work)
        for grad_bucket in moe_grad_buckets:
            work = self.moe_reducer._run_comm_hook(grad_bucket)
            comm_work.append(work)
        for work in comm_work:
            work.wait()

    def _match_unused_params_allreduce(self):
        locally_used_param_map = self.reducer._get_local_used_map()
        self.process_group.allreduce(locally_used_param_map)

        locally_used_param_map = self.moe_reducer._get_local_used_map()
        self.experts_process_group.allreduce(locally_used_param_map)

    def _check_global_requires_backward_grad_sync(self, is_joined_rank):
        if not is_joined_rank and self.require_backward_grad_sync:
            requires_sync_tensor = torch.ones(1, device=self.device)
        else:
            requires_sync_tensor = torch.zeros(1, device=self.device)

        work = dist.all_reduce(requires_sync_tensor, group=self.process_group, async_op=True)
        dist.all_reduce(requires_sync_tensor, group=self.experts_process_group)
        return work

    @property
    def moe_distributed_rank(self):
        return dist.get_rank(self.experts_process_group)

    # TODO moe now have no buffer,so not need to sync
    # 1. _check_and_sync_module_buffers
    # 2. _sync_final_model
