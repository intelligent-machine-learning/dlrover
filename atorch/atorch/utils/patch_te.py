# mypy: ignore-errors
from typing import Any, Dict, Tuple, Union

import torch

from atorch.utils.version import package_version_smaller_than


def te_checkpoint_forward(
    ctx,
    run_function,
    distribute_saved_activations,
    get_cuda_rng_tracker,
    tp_group,
    kwargs: Dict[str, Any],
    *args: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, ...]:
    """Call forward function while saving state to be able to
    redo the computation later."""
    ctx.run_function = run_function
    ctx.distribute_saved_activations = distribute_saved_activations

    from transformer_engine.pytorch.distributed import (
        activation_recompute_forward,
        safely_set_viewless_tensor_data,
        split_tensor_into_1d_equal_chunks,
    )

    # Copy the rng states.
    ctx.fwd_cpu_rng_state = torch.get_rng_state()
    ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
    ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

    with torch.no_grad():
        with activation_recompute_forward(activation_recompute=True, recompute_phase=False):
            outputs = run_function(*args, **kwargs)

    # Divide hidden states across model parallel group and only keep
    # the chunk corresponding to the current rank.
    if distribute_saved_activations:
        ctx.input_0_shape = args[0].data.shape
        safely_set_viewless_tensor_data(
            args[0],
            split_tensor_into_1d_equal_chunks(args[0].data, tp_group, new_buffer=True),
        )

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

    ctx.get_cuda_rng_tracker = get_cuda_rng_tracker
    ctx.tp_group = tp_group
    ctx.kwargs = kwargs

    return outputs


def te_checkpoint_backward(ctx, *args: Tuple[Union[torch.Tensor, None], ...]) -> Tuple[Union[torch.Tensor, None], ...]:
    """Call backward function with activation recomputation."""
    if not torch.autograd._is_checkpoint_valid():
        raise RuntimeError("Checkpointing is not compatible with .grad(), " "please use .backward() if possible")

    from transformer_engine.pytorch.distributed import (
        _set_cuda_rng_state,
        activation_recompute_forward,
        detach_variable,
        gather_split_1d_tensor,
        safely_set_viewless_tensor_data,
    )

    inputs = list(ctx.inputs)
    tensor_indices = ctx.tensor_indices
    tensors = ctx.saved_tensors

    # Fill in inputs with appropriate saved tensors.
    for i, idx in enumerate(tensor_indices):
        inputs[idx] = tensors[i]
    inputs = tuple(inputs)

    get_cuda_rng_tracker = ctx.get_cuda_rng_tracker

    if ctx.distribute_saved_activations:
        safely_set_viewless_tensor_data(
            inputs[0],
            gather_split_1d_tensor(inputs[0].data, ctx.tp_group).view(ctx.input_0_shape),
        )

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
        with activation_recompute_forward(activation_recompute=True, recompute_phase=True):
            outputs = ctx.run_function(*detached_inputs, **ctx.kwargs)

    # Set the states back to what it was at the start of this function.
    torch.set_rng_state(bwd_cpu_rng_state)
    _set_cuda_rng_state(bwd_cuda_rng_state)
    get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

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
    return (None, None, None, None, None) + grads


def patch_te_if_needed():
    # patch checkpoint if te version < 1.3
    if package_version_smaller_than("transformer_engine", "1.3"):
        from transformer_engine.pytorch.distributed import CheckpointFunction

        CheckpointFunction.forward = te_checkpoint_forward
        CheckpointFunction.backward = te_checkpoint_backward
