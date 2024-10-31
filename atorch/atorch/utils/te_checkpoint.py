# This checkpoint implementation is taken from TransformerEngine with modification

from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from torch.cuda import _lazy_call, _lazy_init
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import detach_variable, noop_context_fn

_ALL_ACTIVE_RNG_STATES: Dict = {}


def get_all_rng_states():
    """Returns all generator states used by `CudaRNGStatesTracker`."""
    return _ALL_ACTIVE_RNG_STATES


def set_all_rng_states(states):
    """Updates all generator states used by `CudaRNGStatesTracker`."""
    global _ALL_ACTIVE_RNG_STATES
    _ALL_ACTIVE_RNG_STATES = states


def graph_safe_rng_available():
    """Returns whether cuda graph safe RNG state manipulation is supported."""
    return (
        hasattr(torch.cuda.CUDAGraph, "register_generator_state")
        and hasattr(torch.Generator, "graphsafe_set_state")
        and hasattr(torch.Generator, "graphsafe_get_state")
        and hasattr(torch.Generator, "clone_state")
    )


def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda",
    clone: bool = False,
    graph_safe: bool = True,
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU."""

    _lazy_init()
    if len(torch.cuda.default_generators) == 0:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()
    default_generator = torch.cuda.default_generators[idx]
    if graph_safe_rng_available() and graph_safe:
        if clone:
            # Reference to the cloned generator state
            return default_generator.clone_state()
        # Reference to the current generator state
        return default_generator.graphsafe_get_state()
    return default_generator.get_state()


def _set_cuda_rng_state(
    new_state: torch.Tensor,
    device=-1,
    graph_safe=True,
) -> None:
    """Sets the random number generator state of the current GPU."""

    if len(torch.cuda.default_generators) == 0:
        return

    if device == -1:
        device = torch.device("cuda")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)

    def cb() -> None:
        idx = device.index
        if idx is None:
            idx = torch.cuda.current_device()
        default_generator = torch.cuda.default_generators[idx]
        if graph_safe_rng_available() and graph_safe:
            default_generator.graphsafe_set_state(new_state)
            return
        default_generator.set_state(new_state)

    _lazy_call(cb)


class _CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(
        ctx,
        run_function: Callable,
        get_rng_state_tracker: Union[Callable, None],
        context_fn: Union[Callable, None],
        kwargs: Dict[str, Any],
        *args: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Call forward function while saving state to be able to
        redo the computation later."""
        ctx.run_function = run_function

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            ctx.fwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        if context_fn is not None:
            forward_ctx, recompute_ctx = context_fn()
        else:
            forward_ctx, recompute_ctx = noop_context_fn()
        with torch.no_grad(), forward_ctx:
            outputs = run_function(*args, **kwargs)

        # Store everything.
        ctx.inputs = [arg if not torch.is_tensor(arg) else None for arg in args]
        tensor_inputs = [arg if torch.is_tensor(arg) else None for arg in args]
        ctx.save_for_backward(*tensor_inputs)

        ctx.get_rng_state_tracker = get_rng_state_tracker
        ctx.recompute_ctx = recompute_ctx
        ctx.autocast_setting = torch.is_autocast_enabled()
        ctx.kwargs = kwargs

        return outputs

    @staticmethod
    def backward(ctx, *args: Tuple[Union[torch.Tensor, None], ...]) -> Tuple[Union[torch.Tensor, None], ...]:
        """Call backward function with activation recomputation."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), " "please use .backward() if possible")

        inputs = tuple(t if t is not None else arg for (t, arg) in zip(ctx.saved_tensors, ctx.inputs))

        get_rng_state_tracker = ctx.get_rng_state_tracker

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = _get_cuda_rng_state(graph_safe=False)
        if get_rng_state_tracker is not None:
            bwd_cuda_rng_state_tracker = get_rng_state_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        if ctx.autocast_setting and not torch.is_autocast_enabled():
            # TODO: get autocast dtype. Now asssuming bf16.
            with torch.enable_grad(), ctx.recompute_ctx, torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)
        else:
            with torch.enable_grad(), ctx.recompute_ctx:
                outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state, graph_safe=False)
        if get_rng_state_tracker is not None:
            get_rng_state_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i, output in enumerate(outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True," " this checkpoint() is not necessary")

        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)
        return (None, None, None, None) + grads


class _CheckpointFrame:
    """
    Storage frame for forward RNG states and detached activations from the forward recompute.
    """

    def __init__(self, recompute_fn: Callable, get_rng_state_tracker: Callable):
        self.recompute_fn = recompute_fn
        self.recomputed: List = []
        self.count = 0
        self.get_rng_state_tracker = get_rng_state_tracker
        self.fwd_rng_states = None
        self.bwd_rng_states = None

    def cache_rng_states(self, forward=True):
        """Cache fwd/bwd RNG states in the frame to restore later."""
        rng_states = (
            torch.get_rng_state(),
            _get_cuda_rng_state(graph_safe=False),
        )
        if self.get_rng_state_tracker is not None:
            rng_states += (self.get_rng_state_tracker().get_states(),)

        if forward:
            self.fwd_rng_states = rng_states
        else:
            self.bwd_rng_states = rng_states

    def restore_rng_states(self, forward=True):
        """Restore fwd/bwd RNG states that were previously cached into the frame."""
        if forward:
            rng_states = self.fwd_rng_states
        else:
            rng_states = self.bwd_rng_states

        torch.set_rng_state(rng_states[0])
        _set_cuda_rng_state(rng_states[1], graph_safe=False)
        if self.get_rng_state_tracker is not None:
            self.get_rng_state_tracker().set_states(rng_states[2])


class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):  # pylint: disable=too-few-public-methods
    """torch.autograd hook for packing/unpacking tensors during the activation recompute phase."""

    def __init__(self, frame):
        def pack_hook(x):
            """
            Packing hook for each recomputed activation passed into the `ctx.save_for_backward()`
            call in the forward recomputation.
            """
            frame.recomputed.append(x.detach())
            return x.detach()

        def unpack_hook(x):
            """
            No-op unpack hook that will never be called because the backward pass for the
            forward recomputation is never triggered.
            """
            return x

        super().__init__(pack_hook, unpack_hook)


class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):  # pylint: disable=too-few-public-methods
    """torch.autograd hook for packing/unpacking tensors during the checkpointed forward pass."""

    def __init__(self, frame, args, kwargs):
        def pack_hook(x):
            """
            Packing hook for each tensor passed into `ctx.save_for_backward()` call in the
            forward pass. Since this is the first forward pass, we discard the tensor and instead
            pack a placeholder tensor index into the autograd engine context.
            """
            del x
            idx = frame.count
            frame.count += 1
            frame.autocast_setting = torch.is_autocast_enabled()
            return idx

        def unpack_hook(idx):
            """
            Unpacking hook for each tensor that comes out of the `ctx.saved_tensors` call in the
            backward pass. The first time this is called, the _recomputation_hook will save all the
            activation tensors from `ctx.save_for_backward()` in the forward recomputation into the
            _CheckpointFrame. Subsequent calls will simply return the already recomputed activation
            tensor at the given index of the _CheckpointFrame storage.
            """

            if not frame.recomputed:
                # Store current RNG states in the backward pass
                frame.cache_rng_states(forward=False)

                # Set RNG states to what we saved before the forward pass
                frame.restore_rng_states(forward=True)

                # Recompute the forward pass
                if frame.autocast_setting and not torch.is_autocast_enabled():
                    # TODO: set autocast dtype correctly. Assuming bf16 for now.
                    with _recomputation_hook(frame), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        frame.recompute_fn(*args, **kwargs)
                else:
                    with _recomputation_hook(frame):
                        frame.recompute_fn(*args, **kwargs)

                # Restore RNG states back to the backward pass
                frame.restore_rng_states(forward=False)

            # Return the already recomputed activation tensor at the given index
            activation = frame.recomputed[idx]
            frame.recomputed[idx] = None
            return activation

        super().__init__(pack_hook, unpack_hook)


def checkpoint(
    function: Callable,
    *args: Tuple[torch.Tensor, ...],
    **kwargs,
) -> Tuple[torch.Tensor, ...]:
    """
    Parameters
    ----------
    function: Callable
            pytorch module used to run the forward and backward passes using
            the specified :attr:`args` and :attr:`kwargs`.
    get_rng_state_tracker: `Callable`, default = None
            python callable which returns an instance of :func:`CudaRNGStatesTracker`.
            and `use_reentrant=True`. If `None`, it falls back to the default group.
    use_reentrant : bool, default = True
            perform checkpointing in reentrant mode.
    args : tuple
            tuple of torch tensors for inputs to :attr:`function`.
    kwargs : dict
            dictionary of string keys for keyword arguments to :attr:`function`.
    """
    # Pop out te.distributed.checkpoint() arguments
    use_reentrant = kwargs.pop("use_reentrant", True)
    get_rng_state_tracker = kwargs.pop("get_rng_state_tracker", None)

    # Trigger the native PyTorch checkpoint if the function is not or does not contain a
    # Transformer Engine module.
    context_fn = kwargs.pop("context_fn", noop_context_fn)
    determinism_check = kwargs.pop("determinism_check", "default")
    debug = kwargs.pop("debug", False)

    # Otherwise discard unused te.utils.checkpoint.checkpoint() arguments
    # and execute TE's own checkpointing
    # NOTE: This logic uses the TE checkpoint on all custom callable `function` handles because we
    #       cannot be sure there are no TE modules inside the function. It also means we might run
    #       the TE checkpoint for non-TE modules, so the TE checkpoint has to support a potential
    #       user context function.
    del determinism_check, debug
    if use_reentrant:
        return _CheckpointFunction.apply(
            function,
            get_rng_state_tracker,
            context_fn,
            kwargs,
            *args,
        )

    user_forward_ctx, user_recompute_ctx = context_fn()

    def recompute_fn(*args, **kwargs):
        with torch.autograd.enable_grad(), user_recompute_ctx:
            function(*args, **kwargs)

    # Initialize a new checkpoint frame for each new forward pass.
    new_frame = _CheckpointFrame(
        recompute_fn,
        get_rng_state_tracker,
    )
    new_frame.cache_rng_states(forward=True)

    with _checkpoint_hook(new_frame, args, kwargs), user_forward_ctx:
        out = function(*args, **kwargs)

    return out


class CudaRNGStatesTracker:
    """
    For model parallelism, multiple RNG states need to simultaneously exist in order
    to execute operations in or out of the model parallel region. This class keeps
    track of the various RNG states and provides utility methods to maintain them and
    execute parts of the model under a given RNG setting. Using the `add` method, a
    cuda rng state is initialized based on the input `seed` and is assigned to `name`.
    Later, by forking the rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """
        Set to the initial state (no tracker).
        """
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self) -> Dict[str, torch.Tensor]:
        """
        Get rng states. Copy the dictionary so we have direct pointers
        to the states, not just a pointer to the dictionary.
        """
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states: Dict[str, torch.Tensor]) -> None:
        """
        Set the rng states. For efficiency purposes, we do not
        check the size of seed for compatibility.

        states: Dict[str, torch.Tensor]
               A mapping from string names to RNG states.
        """
        self.states_ = states

    def add(self, name: str, seed: int) -> None:
        """
        Adds a new RNG state.

        name: str
             string identifier for the RNG state.
        seed: int
             PyTorch seed for the RNG state.
        """
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception(f"seed {seed} already exists")
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception(f"cuda rng state {name} already exists")

        if graph_safe_rng_available():
            new_state = _get_cuda_rng_state(clone=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
            # Update global states.
            set_all_rng_states(self.states_)
        else:
            # Get the current rng state.
            orig_rng_state = _get_cuda_rng_state()
            # Set the new state and store it.
            torch.cuda.manual_seed(seed)
            self.states_[name] = _get_cuda_rng_state(clone=True)
            # Reset rng state to what it was.
            _set_cuda_rng_state(orig_rng_state)
            # Update global states.
            set_all_rng_states(self.states_)

    @contextmanager
    def fork(self, name: str = "model-parallel-rng"):
        """
        Fork the cuda rng state, perform operations, and exit with
        the original state.

        name: str
             string identifier for the RNG state.
        """
        # Check if we have added the state
        if name not in self.states_:
            raise Exception(f"cuda rng state {name} is not added")
        # Get the reference to current rng state.
        orig_cuda_rng_state = _get_cuda_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # this is redundant with graph-safe API
            if not graph_safe_rng_available():
                self.states_[name] = _get_cuda_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


def get_te_checkpoint_wrapper_fn(use_reentrant=True):
    _CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()

    def get_cuda_rng_tracker():
        return _CUDA_RNG_STATE_TRACKER

    def _checkpoint_func(m, *args, **kargs):
        return checkpoint(
            m,
            *args,
            get_rng_state_tracker=get_cuda_rng_tracker,
            use_reentrant=use_reentrant,
            **kargs,
        )

    checkpoint_wrapper_fn = partial(checkpoint_wrapper, checkpoint_fn=_checkpoint_func)
    return checkpoint_wrapper_fn
