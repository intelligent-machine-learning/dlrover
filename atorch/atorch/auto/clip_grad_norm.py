import math

import torch
import torch.distributed as dist

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.distributed.distributed import parallel_group
from atorch.utils.version import torch_version


def clip_grad_norm(model, max_norm, norm_type=2, optimizer=None, process_group_name_prefix=""):
    """Clip the gradient norm of all parameters of model that returned by auto_accelerate. The norm is computed over
    all parameters' gradients as viewed as a single vector, and the
    gradients are modified in-place.

    Args:
        model (nn.Module): model returned by auto_accelerate.
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'``
                for infinity norm.
        optimizer (optim.Optimizer): optimizer returned by auto_accelerate.
        process_group_name_prefix (str): The prefix name that is used in `ParallelGroupContextManager`
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if torch_version() >= (1, 12, 1) and torch_version() <= (1, 13, 1):
        from torch.distributed.fsdp.fully_sharded_data_parallel import TrainingState_ as TrainingState
    elif torch_version() >= (2, 0, 0):
        from torch.distributed.fsdp._common_utils import TrainingState
    else:
        TrainingState = None
    counter = getattr(model, "_auto_acc_ctx_counter", 0)
    if counter == 0:
        raise ValueError("model is not returned by auto_accelerate")
    strategy_opt_names = AutoAccelerateContext.strategy_opt_names[counter]
    if (
        "amp_native" in strategy_opt_names
        and hasattr(AutoAccelerateContext, "amp_native_grad_scaler")
        and AutoAccelerateContext.amp_native_grad_scaler.get(counter) is not None
    ):
        # fp16 training needs unscale gradient before clipping, bf16 doesn't
        if optimizer is None:
            raise TypeError(
                "Before cliping gradient norm, gradient values need to be unscaled. Please pass optimizer "
                "when calling `clip_grad_norm`."
            )
        optimizer.unscale_()

    def calculate_grad_norm(parameters, norm_type) -> torch.Tensor:
        r"""Calculate gradient norm of an iterable of parameters.
        Returns:
            Total norm of the parameters (viewed as a single vector).
        """
        parameters = [p for p in parameters if p.grad is not None]

        if len(parameters) == 0:
            return torch.tensor(0.0)
        if norm_type == math.inf:
            local_norm = torch.tensor(max(par.grad.detach().abs().max() for par in parameters))
        else:
            # Compute the norm in full precision no matter what
            local_norm = torch.linalg.vector_norm(
                torch.stack(
                    [torch.linalg.vector_norm(par.grad.detach(), norm_type, dtype=torch.float32) for par in parameters]
                ),
                norm_type,
            )
        local_norm.to(dtype=parameters[0].dtype)
        return local_norm

    parameters = model.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    # The following 3 optimization methods will shard gradients to different ranks.
    use_fsdp = "fsdp" in strategy_opt_names
    use_zero2 = "zero2" in strategy_opt_names
    use_tp = "tensor_parallel" in strategy_opt_names

    data_process_group = parallel_group(process_group_name_prefix + "data")
    tensor_parallel_group = parallel_group(process_group_name_prefix + "tensor")
    if not use_fsdp and not use_zero2 and not use_tp:
        # if gradients are not sharded
        total_norm = calculate_grad_norm(parameters, norm_type)
    elif use_fsdp or use_zero2:
        # using zero optimization
        if torch_version() >= (1, 12, 1) and torch_version() <= (1, 13, 1):
            # _lazy_init, _wait_for_previous_optim_step, _is_root and _assert_state are belong to FSDP.
            model._lazy_init()
            model._wait_for_previous_optim_step()
            assert model._is_root, "clip_grad_norm should only be called on the root (parent) instance"
            model._assert_state(TrainingState.IDLE)
            # compute local norms for each rank
            local_norm = calculate_grad_norm(parameters, norm_type).cuda()
            if norm_type == math.inf:
                total_norm = local_norm
                dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=data_process_group)
                if use_tp:
                    dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=tensor_parallel_group)
            else:
                total_norm = local_norm**norm_type
                dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=data_process_group)
                if use_tp:
                    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=tensor_parallel_group)
                total_norm = total_norm ** (1.0 / norm_type)
        elif torch_version() >= (2, 0, 0):
            import torch.distributed.fsdp._traversal_utils as traversal_utils
            from torch.distributed.fsdp._runtime_utils import _lazy_init

            _lazy_init(model, model)
            if not model._is_root:
                raise RuntimeError("`clip_grad_norm_()` should only be called on the root FSDP instance")
            model._assert_state(TrainingState.IDLE)
            all_no_shard = all(not handle.uses_sharded_strategy for handle in traversal_utils._get_fsdp_handles(model))
            if all_no_shard:
                return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type)
            max_norm = float(max_norm)
            norm_type = float(norm_type)
            sharded_params = set()
            nonsharded_params = set()  # `NO_SHARD` or not FSDP-managed
            grads = []
            for handle in traversal_utils._get_fsdp_handles(model):
                target_set = sharded_params if handle.uses_sharded_strategy else nonsharded_params
                if handle._use_orig_params:
                    for param in handle.flat_param._params:
                        target_set.add(param)
                        if param.grad is not None:
                            grads.append(param.grad)
                else:
                    target_set.add(handle.flat_param)
                    if handle.flat_param.grad is not None:
                        grads.append(handle.flat_param.grad)
            for param in model.parameters():
                not_fsdp_managed = param not in sharded_params and param not in nonsharded_params
                if not_fsdp_managed:
                    nonsharded_params.add(param)
                    if param.grad is not None:
                        grads.append(param.grad)
            # Compute local norms (forced to be in FP32)
            local_sharded_norm = calculate_grad_norm(sharded_params, norm_type).to(model.compute_device)
            local_nonsharded_norm = calculate_grad_norm(nonsharded_params, norm_type).to(model.compute_device)
            # Reconstruct the total gradient norm depending on the norm type
            if norm_type == math.inf:
                total_norm = torch.maximum(local_sharded_norm, local_nonsharded_norm)
                dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=model.process_group)
            else:
                total_norm = local_sharded_norm**norm_type
                dist.all_reduce(total_norm, group=model.process_group)
                # All-reducing the local non-sharded norm would count it an extra
                # world-size-many times
                total_norm += local_nonsharded_norm**norm_type
                total_norm = total_norm ** (1.0 / norm_type)
            if model.cpu_offload.offload_params:
                total_norm = total_norm.cpu()

    elif use_tp and not (use_fsdp or use_zero2):
        # only tensor parallel
        total_norm = calculate_grad_norm(parameters, norm_type).cuda()
        if norm_type == math.inf:
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=tensor_parallel_group)
        else:
            total_norm = local_norm**norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=tensor_parallel_group)
            total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        if p.grad is not None:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return total_norm
