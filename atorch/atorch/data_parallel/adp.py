# coding=utf-8
from __future__ import absolute_import, unicode_literals

from collections import OrderedDict  # noqa: F401
from typing import Any, List

import torch
from fairscale.nn.data_parallel.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
    TrainingState,
    cast_floats_to_right_precision,
    free_storage_,
)
from fairscale.utils.containers import apply_to_tensors
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class AllDataParallel(FullyShardedDataParallel):
    numel_padded_per_param: List

    def __init__(
        self,
        module,
        *args,
        inject_optimizer=False,  # add this than FullyShardedDataParallel
        process_group=None,
        **kwargs,
    ):
        """
        Arguments
            inject_optimizer (bool): inject backward process,
                      self.optimizer.step() will be called when grad `reduce_scatter` finish.
            process_group (torch distributed group): control adp world size such ad MoeLayer
        """
        self.inject_optimizer = inject_optimizer
        self.optimizer = None

        kwargs["process_group"] = process_group
        super().__init__(module=module, *args, **kwargs)

    def set_name(self, name):
        self.origin_name = name

    def extra_repr(self) -> str:
        repr = (
            f"world_size={self.world_size}, "
            f"flatten_parameters={self.flatten_parameters}, "
            f"mixed_precision={self.mixed_precision}, "
            f"reshard_after_forward={self.reshard_after_forward}, "
            f"data_type={[p.dtype for p in self.params]}, "
            f"inject_optimizer={self.inject_optimizer}, "
            f"process_group={self.process_group}, "
            f"group_size={self.process_group.size()}, "
            f"group_rank={self.process_group.rank()}, "
        )
        if self.verbose:
            repr = (
                f"rank={self.rank}, " + repr + f"reshard_after_forward={self.reshard_after_forward}, "
                f"compute_dtype={self.compute_dtype}, "
                f"buffer_dtype={self.buffer_dtype}, "
                f"fp32_reduce_scatter={self.fp32_reduce_scatter}, "
                f"compute_device={self.compute_device}"
                f"move_params_to_cpu={self.move_params_to_cpu}, "
                f"move_grads_to_cpu={self.move_grads_to_cpu}, "
                f"bucket_cap_mb={self.bucket_cap_mb}, "
                f"clear_autocast_cache={self.clear_autocast_cache}"
                f"force_input_to_fp32={self.force_input_to_fp32}"
            )
        return repr

    def _setup_streams(self) -> None:
        """Create streams to overlap data transfer and computation."""
        if len(self._streams) > 0 or not self._is_root:
            return
        super(AllDataParallel, self)._setup_streams()

        for idx, (n, m) in enumerate(self.named_modules()):
            if n != "" and isinstance(m, FullyShardedDataParallel):
                # speed=17.0,  all_gather  post_backward all_gather
                if m.optimizer:
                    m.set_optimizer_stream(
                        2,
                        [
                            m._streams["all_gather"],
                            m._streams["post_backward"],  # this module stream
                        ],
                    )

    def set_optimizer(self, build_optimizer_func, stream_length, streams):
        "using in optimizer.register_overlap_optim"
        params = [p for p in self.parameters()]
        params_group = {"params": params}
        # CAREFUL: optimizer with custom op may not using Stream()
        self.optimizer = build_optimizer_func([params_group], is_zero=True)

        self.set_optimizer_stream(stream_length, streams)
        return self.optimizer

    def set_optimizer_stream(self, stream_length, streams):
        if hasattr(self.optimizer, "optimizer"):  # Float16OptimizerWithFloat16Params
            self.optimizer.optimizer.stream_length = stream_length
            self.optimizer.optimizer.streams = streams
            # Float16OptimizerWithFloat16Params
            self.optimizer.enable_timer = False
            self.optimizer.async_op = True

    def _register_pre_backward_hooks(self, outputs: Any) -> Any:
        """
        after fsdp pre_backward
        add backward_overlap_hook
        """
        outputs = super()._register_pre_backward_hooks(outputs)
        if self.inject_optimizer and self.optimizer:
            grad_acc = outputs.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self.backward_overlap_hook)
        self._overlap_callback_queued = False

        return outputs

    def backward_overlap_hook(self, grad_input, grad_output):
        """Registers a backward hook on the module.

        The hook will be called every time the gradients with respect to module
        inputs are computed. The hook should have the following signature::

            hook(module, grad_input, grad_output) -> tuple(Tensor) or None
        """
        if not self._overlap_callback_queued:
            # First hook callback will see PRE state. If we have multiple params,
            # then subsequent hook callbacks will see POST state.
            self.assert_state([TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST])
            self._overlap_callback_queued = True
            Variable._execution_engine.queue_callback(self.step)

    @torch.no_grad()
    def step(self, *unused_args):
        """
        1. set optimizer params group to self
        2. optimizer step+update
        3. release optimizer params group
        4. release self.params and grad

        normal module: _wait_for_post_backward
        overlap module _finalize_parameters
        """

        assert (
            self.inject_optimizer and self.optimizer
        ), "should call atorch.optim.adam_offload.register_overlap_optim(module,build_optimizer_func) before training"
        # if args.optimizer_overlap == False,will have no optimizer

        # self._require_backward_grad_sync default is True,but in optimizer do
        # not need it
        if self._require_backward_grad_sync:
            # Flush any unreduced buckets in the post_backward stream.
            with torch.cuda.stream(self._streams["post_backward"]):
                assert self._reducer is not None
                self._reducer.flush()
            # wait post_reduction finish,otherwise param.grad will be not None
            torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
            if self.move_grads_to_cpu:
                # Wait for the non-blocking GPU -> CPU grad transfers to
                # finish.
                torch.cuda.current_stream().synchronize()
        # Free reducer buffers.
        if self._reducer is not None:
            self._reducer.teardown()
        # assert param.grad is not None

        def _finalize_parameters(
            fsdp_module: FullyShardedDataParallel,
        ) -> None:
            """Helper used below on all fsdp modules."""
            for p in fsdp_module.params:
                if not p.requires_grad:
                    continue
                if hasattr(p, "_shard_bwd_hook"):
                    assert len(p._shard_bwd_hook) == 2, len(p._shard_bwd_hook)
                    p._shard_bwd_hook[1].remove()
                    delattr(p, "_shard_bwd_hook")

                # Leave the gradient accumulation state as-is if not synchronizing this pass. This ensures p.grad
                # remains the unsharded gradient accumulated from prior no-sync passes, and p._saved_grad_shard
                # remains the sharded gradient from the last synchronized pass. This also allows interleaved no-sync and
                # sync passes, if desired.
                if not self._require_backward_grad_sync:
                    continue
                # Parameter and gradient devices must match.
                if hasattr(p, "_cpu_grad"):
                    # move_params_to_cpu,we have copy params to gpu, so do not
                    # need this assert
                    if not self.move_params_to_cpu:
                        assert p.device == torch.device("cpu")
                    # self.move_grad to gpu, now this grad is on gpu device
                    p.grad = p._cpu_grad
                elif hasattr(p, "_saved_grad_shard"):
                    assert p.device == p._saved_grad_shard.device, "device not equal%s  saved grad:%s" % (
                        p.device,
                        p._saved_grad_shard.device,
                    )
                    p.grad = p._saved_grad_shard
                # release p.data or not

                if hasattr(p, "_saved_grad_shard"):
                    delattr(p, "_saved_grad_shard")

        for m in self.modules():  # includes self
            if isinstance(m, FullyShardedDataParallel):
                # try: #TODO just debug for this function exception
                _finalize_parameters(m)
                # important flag,donot remove
                m._pre_backward_hook_has_run = False
                if any(p.requires_grad for p in m.parameters()):
                    # Check if the module has params and if any of them has
                    # the `requires_grad` field set. If `requires_grad=False` for
                    # all the params, the post_backward hook will not fire and the
                    # state will remain in `TrainingState.BACKWARD_PRE`.
                    if any([p.requires_grad for p in m.params]):
                        m.assert_state(TrainingState.BACKWARD_POST)
                    else:
                        m.assert_state(TrainingState.BACKWARD_PRE)
                else:
                    # When `m` and its children has no params or has params but
                    # none with `requires_grad==True`, there are two cases:
                    # 1. output tensors are `requires_grad==True`. In this case,
                    # pre-backward hook is still registered, so it is in BACKWARD_PRE state.
                    # 2. output tensors are `requires_grad==False`. In this case,
                    # pre-backward hook is not registered, so it is in IDLE
                    # state.
                    m.assert_state([TrainingState.BACKWARD_PRE, TrainingState.IDLE])
                m.training_state = TrainingState.BACKWARD_POST

                if m._is_root:
                    # reset this flag for cases like "one forward pass +
                    # multiple backward passes"
                    self._post_backward_callback_queued = False
                # except Exception:
                # import traceback
                # self._print_r0("exception when _finalize_parameters:%s"%(traceback.format_exc()))
                # raise
        self.training_state = TrainingState.IDLE
        with torch.cuda.stream(self._streams["all_gather"]):
            (
                update_successful,
                grad_norm,
                num_zeros_in_grad,
            ) = self.optimizer.step()
            self._overlap_callback_queued = False

    @torch.no_grad()
    def _wait_for_post_backward(self) -> None:
        """Wait for post-backward to finish. Only called on root instance.
        diff with FSDP: do not call `_finalize_parameters` when param is reject optimizer
        """
        assert self._is_root
        # Check if the root module has params and if any of them has
        # the `requires_grad` field set. If `requires_grad=False` for
        # all the params, the post_backward hook will not fire and the
        # state will remain in `TrainingState.BACKWARD_PRE`.
        if any([p.requires_grad for p in self.params]):
            self.assert_state(TrainingState.BACKWARD_POST)
        else:
            self.assert_state(TrainingState.BACKWARD_PRE)

        if self._require_backward_grad_sync:
            # Flush any unreduced buckets in the post_backward stream.
            with torch.cuda.stream(self._streams["post_backward"]):
                assert self._reducer is not None
                self._reducer.flush()
            torch.cuda.current_stream().wait_stream(self._streams["post_backward"])
            if self.move_grads_to_cpu:
                # Wait for the non-blocking GPU -> CPU grad transfers to
                # finish.
                torch.cuda.current_stream().synchronize()

        # A backward pass is done, clean up below.

        # Free reducer buffers.
        if self._reducer is not None:
            self._reducer.teardown()

        def _finalize_parameters(
            fsdp_module: FullyShardedDataParallel,
        ) -> None:
            """Helper used below on all fsdp modules."""
            for p in fsdp_module.params:
                if not p.requires_grad:
                    continue
                if hasattr(p, "_shard_bwd_hook"):
                    assert len(p._shard_bwd_hook) == 2, len(p._shard_bwd_hook)
                    p._shard_bwd_hook[1].remove()
                    delattr(p, "_shard_bwd_hook")

                # Leave the gradient accumulation state as-is if not synchronizing this pass. This ensures p.grad
                # remains the unsharded gradient accumulated from prior no-sync passes, and p._saved_grad_shard
                # remains the sharded gradient from the last synchronized pass. This also allows interleaved no-sync and
                # sync passes, if desired.
                if not self._require_backward_grad_sync:
                    continue
                # Parameter and gradient devices must match.
                if hasattr(p, "_cpu_grad"):
                    # move_params_to_cpu,we have copy params to gpu, so do not
                    # need this assert
                    if not self.move_params_to_cpu:
                        assert p.device == torch.device("cpu")
                    # self.move_grad to gpu, now this grad is on gpu device
                    p.grad = p._cpu_grad
                elif hasattr(p, "_saved_grad_shard"):
                    assert p.device == p._saved_grad_shard.device, "device not equal%s  saved grad:%s" % (
                        p.device,
                        p._saved_grad_shard.device,
                    )
                    p.grad = p._saved_grad_shard
                # release p.data or not

                if hasattr(p, "_saved_grad_shard"):
                    delattr(p, "_saved_grad_shard")

        # Update root and nested FSDP's hooks and flags.
        for m in self.modules():  # includes self
            if isinstance(m, FullyShardedDataParallel) and m.optimizer is None:
                try:  # TODO just debug for this function exception
                    _finalize_parameters(m)
                except Exception:
                    import traceback

                    self._print_r0("exception when _finalize_parameters:%s" % (traceback.format_exc()))
                    raise
                m._pre_backward_hook_has_run = False
                if any(p.requires_grad for p in m.parameters()):
                    # Check if the module has params and if any of them has
                    # the `requires_grad` field set. If `requires_grad=False` for
                    # all the params, the post_backward hook will not fire and the
                    # state will remain in `TrainingState.BACKWARD_PRE`.
                    if any([p.requires_grad for p in m.params]):
                        m.assert_state([TrainingState.BACKWARD_POST, TrainingState.IDLE])
                    else:
                        m.assert_state([TrainingState.BACKWARD_PRE, TrainingState.IDLE])
                else:
                    # When `m` and its children has no params or has params but
                    # none with `requires_grad==True`, there are two cases:
                    # 1. output tensors are `requires_grad==True`. In this case,
                    # pre-backward hook is still registered, so it is in BACKWARD_PRE state.
                    # 2. output tensors are `requires_grad==False`. In this case,
                    # pre-backward hook is not registered, so it is in IDLE
                    # state.
                    m.assert_state([TrainingState.BACKWARD_PRE, TrainingState.IDLE])
                m.training_state = TrainingState.IDLE

                if m._is_root:
                    # reset this flag for cases like "one forward pass +
                    # multiple backward passes"
                    self._post_backward_callback_queued = False

    def _post_reduction_hook(self, param: Parameter, reduced_grad: torch.Tensor) -> None:
        """Hook to call on each param after the reduce-scatter."""
        assert torch.cuda.current_stream() == self._streams["post_backward"]
        # self.assert_state(TrainingState.BACKWARD_POST)#remove this?
        if self.gradient_postdivide_factor > 1:
            # Average grad by world_size for consistency with PyTorch DDP.
            reduced_grad.data.div_(self.gradient_postdivide_factor)
        # Cast grad to param's dtype (typically FP32). Note: we do this
        # before the move_grads_to_cpu step so that this entire hook remains
        # non-blocking. The downside is a bit more D2H transfer in that case.
        if self.mixed_precision:
            orig_param_grad_data = reduced_grad.data
            reduced_grad.data = reduced_grad.data.to(dtype=param.data.dtype)
            # Don't let this memory get reused until after the transfer.
            orig_param_grad_data.record_stream(torch.cuda.current_stream())

        if param._is_sharded:
            # Accumulate into the gradient shard.
            if getattr(param, "_saved_grad_shard", None) is None:
                param._saved_grad_shard = reduced_grad.data
            else:
                assert (
                    param._saved_grad_shard.shape == reduced_grad.shape
                ), f"{param._saved_grad_shard.shape} vs {reduced_grad.shape}"
                param._saved_grad_shard.data += reduced_grad.data
            reduced_grad = param._saved_grad_shard.data

        # Optionally move gradients to CPU, typically used if one is running the optimizer on the CPU. Once the full
        # backwards pass completes, we will set `.grad` to the CPU copy.
        if self.move_grads_to_cpu:
            param._cpu_grad.copy_(reduced_grad.data, non_blocking=True)
            # Don't let this memory get reused until after the transfer.
            reduced_grad.data.record_stream(torch.cuda.current_stream())

    @torch.no_grad()
    def _shard_parameters_(self) -> None:
        """
        At initialization we wrap a module with full parameters and shard the
        parameters in-place. Sharding is implemented by viewing each parameter
        as a 1D Tensor and retaining only a single slice, where the slice size
        is determined by the number of data parallel workers.

        Wrapping modules with many small parameters (or with a very large data
        parallel world size) will result in many small parameter shards and slow
        performance. In this case it's better to set *``flatten_parameters``* to
        ``True``, so that all of the small parameters in the module are combined
        into a single contiguous Tensor and sharded once.

        After this initial sharding is complete, the user can initialize a
        ``torch.optim.Optimizer`` in the usual way, i.e.::

        .. code-block:: python

            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

        The optimizer will see only a single slice of parameters and will thus
        allocate less memory for optimizer state, avoiding redundancy across
        data parallel workers.
        """
        self.numel_padded_per_param = []
        for p in self.params:
            assert not hasattr(p, "_is_sharded")
            assert p.is_floating_point()
            if self.mixed_precision:
                assert p.dtype == torch.float32, "%s params(%s) is not fp32" % (self, p.dtype)

            # If world_size is 1, then we all-reduce grads instead of sharding.
            p._is_sharded = self.world_size > 1
            p._orig_size = p.data.size()

            if not p._is_sharded:
                p._is_sharded = False
                self.numel_padded_per_param.append(0)
                continue
            p._is_sharded = True

            # Replace p.data with the relevant shard.
            orig_data = p.data
            p.data, num_padded = self._get_shard(p.data)
            self.numel_padded_per_param.append(num_padded)
            free_storage_(orig_data)

            p._is_sharded = True
        assert len(self.numel_padded_per_param) == len(self.params)

    def _clone_params(self, outputs: Any) -> Any:
        param_ptr_set = set(p.storage().data_ptr() for p in self.params)

        def _clone(t: torch.Tensor) -> torch.Tensor:
            if t.storage().data_ptr() in param_ptr_set:
                return t.clone()
            else:
                return t

        outputs = apply_to_tensors(_clone, outputs)
        return outputs

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._lazy_init()

        # Start of a forward pass.
        self.training_state = TrainingState.FORWARD

        # For root and mixed precision, we convert the input to FP16 (no_grad is needed for
        # the conversion).
        if self._is_root and self.mixed_precision:
            args, kwargs = cast_floats_to_right_precision(True, True, *args, **kwargs)

        # If enabled, convert the input to FP32 if we are in full precision.
        # no_grad is not used because the input might be for a non-root instance,
        # which mean autograd needs to go through the conversion.
        if self.force_input_to_fp32 and not self.mixed_precision:
            args, kwargs = cast_floats_to_right_precision(False, False, *args, **kwargs)

        # All-gather full parameters. This will also transfer FP32 parameters to
        # ``self.compute_dtype`` (e.g., FP16 if *mixed_precision* is ``True``).
        # TODO: have no forward timer
        self._rebuild_full_params()

        # self._print_r0("%s rebuild_full_params"%self.origin_name)
        # Register backward hooks to reshard params and reduce-scatter grads.
        # These need to be re-registered every forward pass.
        self._register_post_backward_hooks()

        outputs = self.module(*args, **kwargs)
        outputs = self._clone_params(outputs)  # resolve params release

        if self.reshard_after_forward:
            self._free_full_params()
            if self.mixed_precision:
                self._free_fp16_param_shard()

        # Switch to main FP32 param shard. We maintain this invariant throughout
        # the code, i.e., ``p.data == p._fp32_shard`` after each function. This
        # also ensures that after the first forward, the optimizer state will be
        # initialized with the correct dtype and (sharded) size, since optimizer
        # state is typically initialized lazily in ``optim.step()``.
        self._use_fp32_param_shard()

        # Register pre-backward hooks to all-gather the
        # params for the backward
        # pass (if output's grad was needed).
        # This won't register anything if
        # we are in eval mode.
        #
        # Some model does forward pass multiple times,
        # we need to register the
        # pre-backward hook on every output
        # since the last output's hook has to
        # fire first to setup for backward. However,
        # we use ``self._pre_backward_hook_has_run``
        # to prevent repeated overhead from multiple hook callbacks.
        outputs = self._register_pre_backward_hooks(outputs)
        self._overlap_callback_queued = False
        # Done with a forward pass.
        self.training_state = TrainingState.IDLE

        # Only need to clear cache during forward.
        # During backward, the cache is not used.
        # TODO (Min): Future PyTorch versions may
        # provide a way to completely disable this
        #     cache. Update this when that's available.
        if self.clear_autocast_cache:
            torch.clear_autocast_cache()
        # self._print_r0("%s finish forward"%self.origin_name)

        return outputs

    def get_params_summary(self):
        total_params = sum([p.numel() for p in self.params])
        mean = torch.mean(torch.cat(self.params))
        std = torch.std(torch.cat(self.params))
        return total_params, mean, std

    def _wait_for_previous_optim_step(self) -> None:
        """
        The outer-most :class:`FullyShardedDataParallel` instance (i.e., the root
        instance) needs to synchronize with the default stream to ensure the
        previous optimizer step is done.
        """
        if not torch.cuda.is_available():
            return
        if self.mixed_precision:
            self._streams["fp32_to_fp16"].wait_stream(torch.cuda.current_stream())
        else:
            self._streams["all_gather"].wait_stream(torch.cuda.current_stream())
        self._streams["post_backward"].wait_stream(torch.cuda.current_stream())

    # Since we have overloads above, we can use Any here.
    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._lazy_init()
        self._wait_for_previous_optim_step()
        state_dict = super().state_dict(*args, **kwargs)

        return state_dict
