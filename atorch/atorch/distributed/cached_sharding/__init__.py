import torch
__ACTIVATED = False
__intra_node_pg = None
LEGAL_VERSION = ("3e5bc3f6bb7694221c874b3b10b48b2064c6bd41",)
def activated():
    return __ACTIVATED
def set_activated(b):
    global __ACTIVATED
    __ACTIVATED = b
def legal_version():
    return torch.version.git_version in LEGAL_VERSION
if legal_version():
    # from torch.distributed.fsdp import _init_utils, _runtime_utils, _unshard_param_utils, flat_param
    import torch.distributed.fsdp._init_utils
    import torch.distributed.fsdp._runtime_utils
    import torch.distributed.fsdp._unshard_param_utils
    import torch.distributed.fsdp.flat_param
    from torch.distributed.fsdp._init_utils import (
        HYBRID_SHARDING_STRATEGIES,
        SHARDING_STRATEGY_MAP,
        FlatParamHandle,
        List,
        Optional,
        ProcessGroupType,
        ShardingStrategy,
        _FSDPPolicy,
        _FSDPState,
        _get_default_group,
        _init_process_group_state_for_hybrid_shard,
        nn,
        no_type_check,
    )
    from torch.distributed.fsdp._runtime_utils import (
        RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
        Any,
        HandleTrainingState,
        TrainingState,
        _assert_in_training_states,
        _clear_grads_if_needed,
        _get_handles_to_prefetch,
        _HandlesKey,
        _is_composable,
        _p_assert,
        _PrefetchMode,
        _register_post_backward_final_callback,
    )
    from torch.distributed.fsdp._unshard_param_utils import (
        _module_handles,
        _reshard_grads,
        _unflatten_as_params,
        _unshard_grads,
        _validate_unshard_params_args,
        _writeback_to_local_shard,
        contextlib,
    )
    from torch.distributed.fsdp.flat_param import F, HandleShardingStrategy, Sequence, Tensor, Union, dist, log
    from atorch.distributed.distributed import init_intra_node_process_group, intra_node_process_group
    @no_type_check
    def _init_process_group_state(
        state: _FSDPState,
        process_group: ProcessGroupType,
        sharding_strategy: ShardingStrategy,
        policy: Optional[_FSDPPolicy],
    ) -> _FSDPState:
        if sharding_strategy in HYBRID_SHARDING_STRATEGIES:
            if process_group is None and policy is None:
                # Raise an error here, since this is manual wrapping with no process group
                # passed in, there is no way to ensure all wrapped FSDP instances use the same
                # process groups.
                raise ValueError(
                    f"Manual wrapping with {sharding_strategy} requires explicit specification of process group."
                )
            else:
                state = _init_process_group_state_for_hybrid_shard(state, process_group)
                assert state.process_group is not None, "Expected to populate state.process_group for hybrid shard"
                assert state._inter_node_pg is not None, "Expected to populate state._inter_node_pg for hybrid shard"
                assert (
                    state._inter_node_state is not None
                ), "Expected to populate state._inter_node_state for hybrid shad."
        else:
            state.process_group = process_group if process_group is not None else _get_default_group()
            pg = intra_node_process_group()
            if pg is None:
                pg = init_intra_node_process_group()
            state.intra_node_pg = pg
        state.rank = state.process_group.rank()
        state.world_size = state.process_group.size()
        return state
    @no_type_check
    def _init_param_handle_from_params(
        state: _FSDPState,
        params: List[nn.Parameter],
        fully_sharded_module: nn.Module,
    ):
        if len(params) == 0:
            return
        handle = FlatParamHandle(
            params,
            fully_sharded_module,
            state.compute_device,
            SHARDING_STRATEGY_MAP[state.sharding_strategy],
            state.cpu_offload.offload_params,
            state.mixed_precision.param_dtype,
            state.mixed_precision.reduce_dtype,
            state.mixed_precision.keep_low_precision_grads,
            state.process_group,
            state._use_orig_params,
            state.intra_node_pg,
        )
        # TODO: Can simplify call `shard()` in the `FlatParamHandle` ctor
        handle.shard()
        assert handle not in state._handles
        state.params.append(handle.flat_param)
        state._handles.append(handle)
        state._fully_sharded_module_to_handles[handle._fully_sharded_module].append(handle)
        num_fully_sharded_module_handles = len(state._fully_sharded_module_to_handles[handle._fully_sharded_module])
        assert num_fully_sharded_module_handles == 1, (
            "The current design assumes a module manages at most one "
            f"`FlatParamHandle` but got {num_fully_sharded_module_handles}"
        )
        cpu_device = torch.device("cpu")
        if state.cpu_offload.offload_params and handle.flat_param.device != cpu_device:
            handle.flat_param_to(cpu_device)
    @no_type_check
    def _unshard(
        state: _FSDPState,
        handles: List[FlatParamHandle],
        unshard_stream: torch.cuda.Stream,
        pre_unshard_stream: torch.cuda.Stream,
        post_backward_stream: torch.cuda.Stream,
        hierarchical_sharding: bool,
    ) -> None:
        """
        Unshards the handles in ``handles``. If the handles are in
        :meth:`summon_full_params` and are using mixed precision, then they are
        forced to full precision.
        Postcondition: Each handle's ``FlatParameter`` 's data is the padded
        unsharded flat parameter on the compute device.
        """
        if not handles:
            return
        any_ran_pre_unshard = False
        with torch.cuda.stream(pre_unshard_stream):
            for handle in handles:
                ran_pre_unshard = handle.pre_unshard()
                any_ran_pre_unshard = any_ran_pre_unshard or ran_pre_unshard
        if any_ran_pre_unshard:
            unshard_stream.wait_stream(pre_unshard_stream)
        if state.limit_all_gathers:
            event = state._free_event_queue.dequeue_if_needed()
            if event:
                event.synchronize()
        with torch.cuda.stream(unshard_stream):
            for handle in handles:
                free_unsharded_flat_param = (
                    not state._is_root and handle._sharding_strategy in RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES
                )
                handle.unshard(free_unsharded_flat_param, post_backward_stream, hierarchical_sharding)
                handle.post_unshard()
    @no_type_check
    def _reshard(
        state: _FSDPState,
        handles: List[FlatParamHandle],
        free_unsharded_flat_params: List[bool],
    ):
        """
        Reshards the handles in ``handles``. ``free_unsharded_flat_params`` should
        have the same length as ``handles``, and each element should give whether
        the corresponding handle should free its padded unsharded flat parameter.
        """
        if not handles:
            return
        _p_assert(
            len(handles) == len(free_unsharded_flat_params),
            "Expects both lists to have equal length but got " f"{len(handles)} and {len(free_unsharded_flat_params)}",
        )
        post_backward_stream = state._streams["post_backward"]
        for handle, free_unsharded_flat_param in zip(
            handles,
            free_unsharded_flat_params,
        ):
            handle.reshard(free_unsharded_flat_param, post_backward_stream)
            if state.limit_all_gathers and free_unsharded_flat_param:
                free_event = torch.cuda.Event()
                free_event.record()
                state._free_event_queue.enqueue(free_event)
            handle.post_reshard()
        # Since we prefetch entire handles keys at a time, conservatively mark
        # the entire key as no longer prefetched once we free at least one
        handles_key = tuple(handles)
        if any(free_unsharded_flat_params):
            state._handles_prefetched.pop(handles_key, None)
    @no_type_check
    def _pre_forward_unshard(
        state: _FSDPState,
        handles: List[FlatParamHandle],
    ) -> None:
        """Unshards parameters in the pre-forward."""
        if not handles:
            return
        handles_key = tuple(handles)
        # If the handles have been prefetched, then there is no need to call
        # `_unshard()` again
        if not state._handles_prefetched.get(handles_key, False):
            _unshard(
                state,
                handles,
                state._streams["unshard"],
                state._streams["pre_unshard"],
                state._streams["post_backward"],
                hierarchical_sharding=True,
            )
        state._needs_pre_forward_unshard[handles_key] = False
        torch.cuda.current_stream().wait_stream(state._streams["unshard"])
        _prefetch_handles(state, handles_key, _PrefetchMode.FORWARD)
    @no_type_check
    def _pre_backward_hook(
        state: _FSDPState,
        module: nn.Module,
        _handles: List[FlatParamHandle],
        *unused: Any,
    ) -> Any:
        """
        Prepares ``_handles`` 's ``FlatParameter`` s for gradient computation.
        Args:
            module (nn.Module): Fully sharded module (see [Note: Fully Sharded
                Module]).
        """
        _handles_key = tuple(_handles)  # avoid shadowing `handles_key`
        # Only run the pre-backward hook once per group of handles involved in the
        # same module forward computation
        if _handles_key and state._ran_pre_backward_hook.get(_handles_key, False):
            return
        with torch.profiler.record_function("FullyShardedDataParallel._pre_backward_hook"):
            # Queue the post-backward callback once for the root FSDP instance to
            # attach it to the outermost backward graph task so that it is called
            # after all backward calls complete
            if state._is_root and not state._post_backward_callback_queued:
                _register_post_backward_final_callback(state, module)
                _clear_grads_if_needed(state._all_handles)
            elif _handles_key:
                allowed_states = [TrainingState.IDLE]
                if _is_composable(state):
                    allowed_states.append(TrainingState.FORWARD_BACKWARD)
                _assert_in_training_states(state, allowed_states)
            state.training_state = TrainingState.FORWARD_BACKWARD
            # Queueing the post-backward callback is the only logic that is not
            # per-handle in the pre-backward hook, so we can return early here if
            # there are no handles.
            if not _handles_key:
                return
            for handle in _handles:
                handle._training_state = HandleTrainingState.BACKWARD_PRE
            if state._needs_pre_backward_unshard[_handles_key]:
                # If the handles have been prefetched, then there is no need to
                # call `_unshard()` again
                if not state._handles_prefetched.get(_handles_key, False):
                    _unshard(
                        state,
                        _handles,
                        state._streams["unshard"],
                        state._streams["pre_unshard"],
                        state._streams["post_backward"],
                        hierarchical_sharding=True,
                    )
                torch.cuda.current_stream().wait_stream(state._streams["unshard"])
            # Set this to `False` to ensure that a mistargeted prefetch does not
            # actually unshard these handles
            state._needs_pre_backward_unshard[_handles_key] = False
            _prefetch_handles(state, _handles_key, _PrefetchMode.BACKWARD)
            for handle in _handles:
                handle.prepare_gradient_for_backward()
            state._ran_pre_backward_hook[_handles_key] = True
    @no_type_check
    def _prefetch_handles(
        state: _FSDPState,
        current_handles_key: _HandlesKey,
        prefetch_mode: _PrefetchMode,
    ) -> None:
        """
        Prefetches the next handles if needed (without synchronization). An empty
        handles key cannot prefetch.
        """
        if not current_handles_key:
            return
        handles_to_prefetch = _get_handles_to_prefetch(state, current_handles_key)
        for handles_key in handles_to_prefetch:
            # Temporarily emulate the training state while calling `_unshard` to
            # ensure the correct `as_params` for `_use_unsharded_views()`
            prev_training_states: List[HandleTrainingState] = []
            for handle in handles_key:
                prev_training_states.append(handle._training_state)
                if prefetch_mode == _PrefetchMode.BACKWARD:
                    handle._training_state = HandleTrainingState.BACKWARD_PRE
                elif prefetch_mode == _PrefetchMode.FORWARD:
                    handle._training_state = HandleTrainingState.FORWARD
                else:
                    raise ValueError(f"Invalid prefetch mode on rank {state.rank}: {prefetch_mode}")
            # Prefetch the next set of handles without synchronizing to allow
            # the sync to happen as late as possible to maximize overlap
            _unshard(
                state,
                handles_key,
                state._streams["unshard"],
                state._streams["pre_unshard"],
                state._streams["post_backward"],
                hierarchical_sharding=True,
            )
            for handle, prev_training_state in zip(handles_key, prev_training_states):
                handle._training_state = prev_training_state
            state._handles_prefetched[handles_key] = True
    @contextlib.contextmanager
    def _unshard_fsdp_state_params(
        module: nn.Module,
        state: _FSDPState,
        writeback: bool,
        rank0_only: bool,
        offload_to_cpu: bool,
        with_grads: bool,
    ):
        """
        This unshards the parameters for a single FSDP state ``state`` that
        corresponds to ``module``.
        """
        _validate_unshard_params_args(state, writeback, rank0_only, offload_to_cpu, with_grads)
        torch.cuda.synchronize()
        # If handles are shared by other module(s), the handle may be already unsharded.
        handles = [
            handle
            for handle in _module_handles(state, module)
            if handle._training_state != HandleTrainingState.SUMMON_FULL_PARAMS
        ]
        if not handles:
            yield
            return
        for handle in handles:
            assert (
                handle._training_state == HandleTrainingState.IDLE
            ), f"Expects the handle training to be IDLE but got {handle._training_state}"
        for handle in handles:
            handle._training_state = HandleTrainingState.SUMMON_FULL_PARAMS
        _clear_grads_if_needed(handles)
        free_unsharded_flat_params = [handle.needs_unshard() for handle in handles]
        # No need to call `wait_stream()` since we unshard in the computation
        # stream directly
        computation_stream = torch.cuda.current_stream()
        _unshard(
            state, handles, computation_stream, computation_stream, computation_stream, hierarchical_sharding=False
        )
        if with_grads:
            _unshard_grads(handles)
        if rank0_only and state.rank != 0:
            # Free the unsharded flattened parameter early
            _reshard(state, handles, free_unsharded_flat_params)
            if with_grads:
                _reshard_grads(handles)
            try:
                yield
            finally:
                for handle in handles:
                    handle._training_state = HandleTrainingState.IDLE
        else:
            # Unflatten the unsharded flattened parameters
            with contextlib.ExitStack() as stack:
                # Invariant: rank == 0 or !rank0_only
                for handle in handles:
                    if offload_to_cpu and handle.uses_sharded_strategy:
                        stack.enter_context(handle.to_cpu())
                        # NOTE: Since PyTorch enforces that a parameter and its
                        # gradients need to match metadata (e.g. device), we must
                        # move gradients to CPU *after* we move parameters.
                # NOTE: This assumes 1 `FlatParameter`
                if not state._use_orig_params:
                    stack.enter_context(_unflatten_as_params(state, module))
                try:
                    yield
                finally:
                    stack.close()
                    if writeback:
                        _writeback_to_local_shard(handles, with_grads)
                    _reshard(state, handles, free_unsharded_flat_params)
                    if with_grads:
                        _reshard_grads(handles)
                    for handle in handles:
                        handle._training_state = HandleTrainingState.IDLE
    class FlatParamHandleHook(FlatParamHandle):
        origin_init = FlatParamHandle.__init__
        def __init__(
            self,
            params: Sequence[Union[nn.Parameter, Tensor]],
            fully_sharded_module: nn.Module,
            device: torch.device,
            sharding_strategy: HandleShardingStrategy,
            offload_params: bool,
            mp_param_dtype: Optional[torch.dtype],
            mp_reduce_dtype: Optional[torch.dtype],
            keep_low_precision_grads: bool,
            process_group: dist.ProcessGroup,
            use_orig_params: bool,
            intra_node_pg: dist.ProcessGroup,
        ):
            FlatParamHandleHook.origin_init(
                self,
                params,
                fully_sharded_module,
                device,
                sharding_strategy,
                offload_params,
                mp_param_dtype,
                mp_reduce_dtype,
                keep_low_precision_grads,
                process_group,
                use_orig_params,
            )
            self.intra_node_pg = intra_node_pg
            self.intra_node_pg_size = intra_node_pg.size()
            self.local_rank = self.rank % self.intra_node_pg_size
        def unshard(self, free_unsharded_flat_param, post_backward_stream, hierarchical_sharding):
            """
            Runs the unshard logic. This includes all-gathering the flat parameter
            and switching to using the unsharded flat parameter. If the handle does
            not need unsharding, then this only switches to using the unsharded
            flat parameter. For ``NO_SHARD``, this is a no-op.
            If FSDP is in :meth:`summon_full_params` and the handle uses parameter
            mixed precision, then the parameter is forced to full precision.
            """
            if not self.needs_unshard():
                # Even when not needing an unshard, we should switch to using
                # the unsharded flat parameter
                unsharded_flat_param = (
                    self._get_padded_unsharded_flat_param() if self.uses_sharded_strategy else self.flat_param
                )
                self._use_unsharded_flat_param(unsharded_flat_param)
                return
            def memory_offloading(unsharded_flat_param_to_offloading):
                # offlading to memory
                chunk, numel_to_pad = FlatParamHandle._get_unpadded_shard(
                    unsharded_flat_param_to_offloading, self.local_rank, self.intra_node_pg_size
                )
                # shard = chunk.clone()
                shard = chunk
                if numel_to_pad > 0:
                    shard = F.pad(chunk, [0, numel_to_pad])
                # create offloading tensor if needed
                if getattr(self.flat_param, "offloading_tensor", None) is None:
                    shard_size = chunk.numel() + numel_to_pad
                    self.flat_param.offloading_tensor = torch.empty(
                        shard_size, dtype=unsharded_flat_param.dtype, device="cpu"
                    )
                    self.flat_param.offloading_tensor = self.flat_param.offloading_tensor.pin_memory()
                    self.flat_param.offloading_count = 0
                # copy to cpu
                post_backward_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(post_backward_stream):
                    self.flat_param.offloading_tensor.copy_(shard, non_blocking=True)
                    self.flat_param.offloading_count += 1
            def memory_reloading(free):
                # loading from memory
                if not free:
                    torch.cuda.current_stream().wait_stream(post_backward_stream)
                unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
                _p_assert(
                    hasattr(self, "intra_node_pg") and hasattr(self, "intra_node_pg_size"),
                    "Expects a intra-node process group and intra_node_pg_size to have been set via `shard()`",
                )
                sharded_flat_param = self.flat_param.offloading_tensor.to(self.device, non_blocking=True)
                expected_numel = sharded_flat_param.numel() * self.intra_node_pg_size
                _p_assert(
                    unsharded_flat_param.numel() == expected_numel,
                    f"Expects {expected_numel} numel but got {unsharded_flat_param.numel()}",
                )
                dist.all_gather_into_tensor(
                    unsharded_flat_param,
                    sharded_flat_param,
                    self.intra_node_pg,
                )
                self._use_unsharded_flat_param(unsharded_flat_param)
                torch.cuda.current_stream().wait_stream(post_backward_stream)
                self.flat_param.offloading_count -= 1
            if (
                self._training_state
                in (
                    HandleTrainingState.BACKWARD_PRE,
                    HandleTrainingState.BACKWARD_POST,
                    HandleTrainingState.IDLE,  # backward prefetch
                )
                and hierarchical_sharding
            ):
                if self.flat_param.offloading_count > 0:
                    # loading from memory
                    memory_reloading(free_unsharded_flat_param)
                else:
                    # loading from network
                    unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
                    padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
                    self._use_unsharded_flat_param(padded_unsharded_flat_param)
                    if free_unsharded_flat_param:
                        memory_offloading(padded_unsharded_flat_param)
            elif self._training_state in (HandleTrainingState.FORWARD,) and hierarchical_sharding:
                # loading from network
                unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
                padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
                self._use_unsharded_flat_param(padded_unsharded_flat_param)
                if free_unsharded_flat_param:
                    memory_offloading(padded_unsharded_flat_param)
            else:
                # loading from network
                unsharded_flat_param = self._alloc_padded_unsharded_flat_param()
                padded_unsharded_flat_param = self._all_gather_flat_param(unsharded_flat_param)
                self._use_unsharded_flat_param(padded_unsharded_flat_param)
        def reshard(self, free_unsharded_flat_param: bool, post_backward_stream: torch.cuda.Stream):
            """
            Runs the reshard logic. This includes freeing the unsharded flat
            parameter if ``free_unsharded_flat_param`` and switching to using the
            sharded flat parameter. Note that this also implicitly offloads
            the sharded flat parameter (if CPU offload is enabled) by pointing
            it to the ``_local_shard`` attribute which resides on CPU.
            """
            # Switch to the sharded `FlatParameter` before freeing to prevent
            # "use-after-free"-type bugs with external profiling tools, where for
            # `use_orig_params=True`, the `param` does not point to valid memory
            # when setting `param.data = ...` in `_use_sharded_views()`.
            self._use_sharded_flat_param()
            if free_unsharded_flat_param:
                if self._training_state == HandleTrainingState.FORWARD:
                    torch.cuda.current_stream().wait_stream(post_backward_stream)
                self._free_unsharded_flat_param()
    log.info("pytorch version(%s) is compatible, Enable cached sharding!", torch.version.git_version)
    backup_funcs = (
        torch.distributed.fsdp._init_utils._init_process_group_state,
        torch.distributed.fsdp._init_utils._init_param_handle_from_params,
        torch.distributed.fsdp._runtime_utils._unshard,
        torch.distributed.fsdp._runtime_utils._reshard,
        torch.distributed.fsdp._runtime_utils._pre_forward_unshard,
        torch.distributed.fsdp._runtime_utils._pre_backward_hook,
        torch.distributed.fsdp._runtime_utils._prefetch_handles,
        torch.distributed.fsdp._unshard_param_utils._unshard_fsdp_state_params,
        torch.distributed.fsdp.flat_param.FlatParamHandle.__init__,
        torch.distributed.fsdp.flat_param.FlatParamHandle.unshard,
        torch.distributed.fsdp.flat_param.FlatParamHandle.reshard,
    )
    replaced_funcs = (
        _init_process_group_state,
        _init_param_handle_from_params,
        _unshard,
        _reshard,
        _pre_forward_unshard,
        _pre_backward_hook,
        _prefetch_handles,
        _unshard_fsdp_state_params,
        FlatParamHandleHook.__init__,
        FlatParamHandleHook.unshard,
        FlatParamHandleHook.reshard,
    )
    def set():
        if activated():
            return
        (
            torch.distributed.fsdp._init_utils._init_process_group_state,
            torch.distributed.fsdp._init_utils._init_param_handle_from_params,
            torch.distributed.fsdp._runtime_utils._unshard,
            torch.distributed.fsdp._runtime_utils._reshard,
            torch.distributed.fsdp._runtime_utils._pre_forward_unshard,
            torch.distributed.fsdp._runtime_utils._pre_backward_hook,
            torch.distributed.fsdp._runtime_utils._prefetch_handles,
            torch.distributed.fsdp._unshard_param_utils._unshard_fsdp_state_params,
            torch.distributed.fsdp.flat_param.FlatParamHandle.__init__,
            torch.distributed.fsdp.flat_param.FlatParamHandle.unshard,
            torch.distributed.fsdp.flat_param.FlatParamHandle.reshard,
        ) = replaced_funcs
        log.warning("pytorch version(%s) is compatible, enable cached sharding!", torch.version.git_version)
        set_activated(True)
    def reset():
        if not activated():
            return
        (
            torch.distributed.fsdp._init_utils._init_process_group_state,
            torch.distributed.fsdp._init_utils._init_param_handle_from_params,
            torch.distributed.fsdp._runtime_utils._unshard,
            torch.distributed.fsdp._runtime_utils._reshard,
            torch.distributed.fsdp._runtime_utils._pre_forward_unshard,
            torch.distributed.fsdp._runtime_utils._pre_backward_hook,
            torch.distributed.fsdp._runtime_utils._prefetch_handles,
            torch.distributed.fsdp._unshard_param_utils._unshard_fsdp_state_params,
            torch.distributed.fsdp.flat_param.FlatParamHandle.__init__,
            torch.distributed.fsdp.flat_param.FlatParamHandle.unshard,
            torch.distributed.fsdp.flat_param.FlatParamHandle.reshard,
        ) = backup_funcs
        log.warning("pytorch version(%s) is compatible, disable cached sharding!", torch.version.git_version)
        set_activated(False)
else:
    def set():
        pass
    def reset():
        pass