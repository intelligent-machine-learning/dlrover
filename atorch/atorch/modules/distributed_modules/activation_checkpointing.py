"""Then implementation of distributed activation checkpointing is taken from Megatron-Lm,
with necessary modification for compatibility.
In essence this does 2 parts:
    1. RNG state tracking: as per torch/utils/checkpoint.py's comment:
        `# We can't know if the run_fn will internally move some args to different devices,
        # which would require logic to preserve rng states for those devices as well.
        # We could paranoically stash and restore ALL the rng states for all visible devices,
        # but that seems very wasteful for most cases.  Compromise:  Stash the RNG state for
        # the device of all Tensor args.`
        Here Megatron's custom implementation of this functionality is adapted.
    2 Checkpoint: For megatron style tensor parallel modules, activations are replicated across the
        tensor parallel group. Storing the full activation is a waste of CUDA memory. A wiser way is to
        save only a portion of the activations and all-gather them in the backward pass.
"""

import contextlib

import torch
from torch import _C
from torch.cuda import _lazy_call
from torch.cuda import device as device_ctx_manager
from torch.utils.checkpoint import detach_variable

from atorch.distributed.distributed import parallel_group_size
from atorch.utils.graph_transform_utils import _pack_kwargs, _unpack_kwargs, combine_map_aggregate, map_aggregate
from atorch.utils.version import torch_version

try:
    if torch_version() <= (2, 0, 0):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper as TorchWrapper
    else:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper as TorchWrapper
except ImportError:
    from atorch.common.log_utils import default_logger as logger

    logger.info("Checkpoint not supported, ignore")
    TorchWrapper = object

from .mappings import gather_split_1d_tensor, split_tensor_into_1d_equal_chunks
from .utils import safely_set_viewless_tensor_data

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        # Map from a string name to the cuda rng state.
        self.states_ = {}
        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def reset(self):
        """Set to the initial state (no tracker)."""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception("seed {} already exists".format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception("cuda rng state {} already exists".format(name))
        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception("cuda rng state {} is not added".format(name))
        # Store current rng state.
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Update the current rng state for later use.
            self.states_[name] = torch.cuda.get_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed, group="tensor"):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-tensor-model-parallel regions.
        tensor-model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    # 2718 is just for fun and any POSITIVE value will work.
    offset = seed + 2718
    tensor_model_parallel_seed = offset + parallel_group_size(group)
    # Data parallel gets the original seed.
    data_parallel_seed = seed

    _CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    # and model parallel state.
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
    two main changes:
        1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
        2) the states in the model parallel tracker are also properly
           tracked/set/reset.
    """

    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            tp_size = parallel_group_size("tensor")
            # Split tensor of rank > 1, and whose first dim is shardable
            # Testing for args[0] is a HACK. The purpose is to avoid sharding inputs that are
            # referenced by multiple nodes. The shard method is an inplace operation, sharding
            # inputs such as attention_mask is illegal for following layers.
            # FIXME needs a smarter way to figure out which one of the inputs is shardable

            def shape_fn(input_):
                if (
                    isinstance(input_, torch.Tensor)
                    and len(input_.data.shape) > 1
                    and input_.data.shape[0] % tp_size == 0
                    and input_ is args[0]
                ):
                    return input_.data.shape
                else:
                    return None

            def split_fn(input_):
                if (
                    isinstance(input_, torch.Tensor)
                    and len(input_.data.shape) > 1
                    and input_.data.shape[0] % tp_size == 0
                    and input_ is args[0]
                ):
                    safely_set_viewless_tensor_data(
                        input_, split_tensor_into_1d_equal_chunks(input_.data, new_buffer=True)
                    )
                return input_

            # None for tensors not sharded
            input_shapes = map_aggregate(args, shape_fn)
            args = map_aggregate(args, split_fn)
            ctx.input_shapes = input_shapes

        # Store everything.
        ctx.inputs = []
        ctx.tensor_indices = []

        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), " "please use .backward() if possible")
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        # Fill in inputs with appropriate saved tensors.
        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        if ctx.distribute_saved_activations:
            input_shapes = ctx.input_shapes
            # Turn the inputs into a tuple (aligning with input_shapes)
            inputs = (*inputs,)

            def _gather_fn(input_, input_shape):
                if input_shape is not None:
                    safely_set_viewless_tensor_data(input_, gather_split_1d_tensor(input_.data).view(input_shape))
                return input_

            inputs = combine_map_aggregate(inputs, input_shapes, _gather_fn)

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True," " this checkpoint() is not necessary")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, distribute_saved_activations, *args):
    """Checkpoint a model or part of the model.
    This has been directly copied from torch.utils.checkpoint.
    TP checkpoint implementation supports state preservance only,so kwargs should have no effect
    """
    return CheckpointFunction.apply(function, distribute_saved_activations, *args)


class TPCheckpointWrapper(TorchWrapper):
    def __init__(self, mod, checkpoint_fn):
        super().__init__(mod)
        self.checkpoint_fn = checkpoint_fn

    def forward(self, *args, **kwargs):
        flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)

        def my_function(*inputs):
            # unpack back into args and kwargs
            unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
            # run original module
            return self._checkpoint_wrapped_module(*unpacked_args, **unpacked_kwargs)

        return self.checkpoint_fn(my_function, True, *flat_args)


def tp_wrap_fn(module):
    return TPCheckpointWrapper(module, checkpoint)
