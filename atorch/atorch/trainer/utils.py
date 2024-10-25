import os
import pickle
from datetime import datetime
from enum import Enum
from typing import List, Optional, Union

import amp_C
import torch
import torch.distributed
from apex.multi_tensor_apply import multi_tensor_applier

import atorch
from atorch.common.log_utils import default_logger as logger
from atorch.utils.import_util import is_megatron_lm_available

if is_megatron_lm_available():
    from megatron.core import mpu, tensor_parallel
    from megatron.core.optimizer import (
        ChainedOptimizer,
        DistributedOptimizer,
        Float16OptimizerWithFloat16Params,
        MegatronOptimizer,
    )
    from megatron.core.transformer.module import MegatronModule, param_is_not_shared
    from megatron.core.transformer.moe.moe_utils import track_moe_metrics
    from megatron.training.global_vars import (
        get_args,
        get_num_microbatches,
        get_one_logger,
        get_tensorboard_writer,
        get_timers,
        get_wandb_writer,
    )
    from megatron.training.theoretical_memory_usage import report_theoretical_memory
    from megatron.training.training import num_floating_point_operations
    from megatron.training.utils import print_rank_last, report_memory


class DistributedType(str, Enum):
    NO_DISTRIBUTE = "NO_DISTRIBUTE"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"  # noqa: F811
    MEGATRON = "MEGATRON"
    MINDSPEED = "MINDSPEED"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    MULTI_NPU = "MULTI_NPU"
    MULTI_MLU = "MULTI_MLU"
    MULTI_MUSA = "MULTI_MUSA"
    MULTI_XPU = "MULTI_XPU"


def is_main_process():
    return atorch.rank() == 0


def is_local_main_process():
    return atorch.local_rank() == 0


def is_last_rank_on_current_stage():
    data_with_cp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    tp_rank = mpu.get_tensor_model_parallel_rank()
    data_last_rank = mpu.get_data_parallel_world_size(with_context_parallel=True) - 1
    tp_last_rank = mpu.get_tensor_model_parallel_world_size() - 1
    return data_with_cp_rank == data_last_rank and tp_rank == tp_last_rank


def dict_to_tensor(input_dict):
    serialized_dict = pickle.dumps(input_dict)
    tensor = torch.tensor(list(serialized_dict), dtype=torch.uint8, device=torch.cuda.current_device())
    return tensor


def tensor_to_dict(tensor):
    serialized_dict = bytes(tensor.tolist())
    return pickle.loads(serialized_dict)


def gather_param_in_tensor_model_group(name, param):
    if param.partition_dim == 0:
        world_size = mpu.get_tensor_model_parallel_world_size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return param

        dim_size = list(param.size())
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=param.dtype, device=torch.cuda.current_device())
        torch.distributed._all_gather_base(output, param.contiguous(), group=mpu.get_tensor_model_parallel_group())

        return output
    elif param.partition_dim == 1:
        world_size = mpu.get_tensor_model_parallel_world_size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return param

        # Size and dimension.
        rank = mpu.get_tensor_model_parallel_rank()

        tensor_list = [torch.empty_like(param) for _ in range(world_size)]
        tensor_list[rank] = param.detach()
        torch.distributed.all_gather(tensor_list, param, group=mpu.get_tensor_model_parallel_group())

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=param.partition_dim).contiguous()

        return output
    else:
        raise ValueError(
            f"{name}.partition_dim: {param.partition_dim} is not valid, "
            "please check whether the param's attribute 'partition_dim' is set correctly."
        )


def distributed_std(name, param):
    if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
        # TODO(@jinshi.cl): consider whether allgather params is necessary.
        param = gather_param_in_tensor_model_group(name, param)
    count = torch.tensor(torch.numel(param), dtype=torch.int32, device=param.device)
    sums = torch.sum(param)
    torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    mean = sums / count
    square_sums = torch.sum(torch.square(param - mean))
    torch.distributed.all_reduce(square_sums, op=torch.distributed.ReduceOp.SUM, group=mpu.get_data_parallel_group())

    return (square_sums.float() / (count - 1)) ** 0.5  # unbiased estimator


def calc_params_std(model, gather_to_last_rank=False):
    """
    Calculate std of parameters.

    compute std for first, middle and last layer
    monitor params:
    word_embeddings.weight, lm_head.weight
    mlp.xxx, attention.dense.weight,
    attention.query_key_value.weight,
    input_layernorm.weight,
    post_attention_layernorm.weight,
    final_layernorm.weight

    """

    args = get_args()
    if not isinstance(model, list):
        model = [model]

    params_std = {}
    logging_layers = [0, args.num_layers // 2, args.num_layers - 1]
    keep_keys = [
        "mlp",
        "self_attention.linear_qkv.layer_norm_weight",  # megatron mcore
        "self_attention.linear_proj.weight",  # megatron mcore
        "self_attention.linear_qkv.weight",  # megatron mcore
        "pre_mlp_layernorm.weight",  # megatron mcore
        "input_layernorm.weight",  # megatron legacy
        "self_attention.dense.weight",  # megatron legacy
        "self_attention.query_key_value.weight",  # megatron legacy
        "post_attention_norm.weight",  # megatron legacy
    ]
    extra_keys = [
        "word_embeddings.weight",
        "output_layer.weight",
        "final_layernorm.weight",  # megatron mcore
        "final_norm.weight",  # megatron legacy
    ]

    def recursive_getattr(obj, keys: list):
        assert len(keys) > 0
        if len(keys) == 1:
            return getattr(obj, keys[0])
        else:
            return recursive_getattr(getattr(obj, keys[0]), keys[1:])

    for _model in model:
        for name, param in _model.named_parameters():
            if any(key in name for key in extra_keys):
                params_std[name] = distributed_std(name, param)
            else:
                # other weights
                for key in keep_keys:
                    if key in name and "bias" not in name:
                        parts = name.split(".")

                        layers_pos_in_parts = parts.index("layers")

                        # layer_idx_in_pp_context is layer index in pipeline model parallel context.
                        layer_idx_in_pp_context = int(parts[layers_pos_in_parts + 1])

                        # The operation of "xxx.xxx.xxx.layer_number - 1" is to ensure the first layer index is 0
                        # actual_layer_idx: is between [0, num_layers)
                        actual_layer_idx = (
                            recursive_getattr(_model, parts[: layers_pos_in_parts + 1])[
                                layer_idx_in_pp_context
                            ].layer_number
                            - 1
                        )

                        if actual_layer_idx in logging_layers:
                            parts[layers_pos_in_parts + 1] = str(actual_layer_idx)
                            real_name = ".".join(parts)
                            params_std[real_name] = distributed_std(name, param)

    if gather_to_last_rank and mpu.get_pipeline_model_parallel_world_size() > 1:
        # Gather scattered metrics on all pipeline stages.
        rank = torch.distributed.get_rank()

        if is_last_rank_on_current_stage():

            max_size_dict = 1024 * 1024
            pp_world_size = mpu.get_pipeline_model_parallel_world_size()
            last_pp_rank = mpu.get_pipeline_model_parallel_last_rank()

            local_tensor = dict_to_tensor(params_std)
            local_tensor_padded = torch.cat(
                [
                    local_tensor,
                    torch.zeros(
                        max_size_dict - len(local_tensor), dtype=torch.uint8, device=torch.cuda.current_device()
                    ),
                ]
            )

            if rank == last_pp_rank:
                gathered_tensors = [
                    torch.empty(max_size_dict, dtype=torch.uint8, device=torch.cuda.current_device())
                    for _ in range(pp_world_size)
                ]
            else:
                gathered_tensors = None

            torch.distributed.gather(
                local_tensor_padded, gathered_tensors, dst=last_pp_rank, group=mpu.get_pipeline_model_parallel_group()
            )

            if rank == last_pp_rank:
                all_dicts = {}
                for tensor in gathered_tensors:
                    received_dict = tensor_to_dict(tensor)
                    all_dicts.update(received_dict)
                params_std = all_dicts
    torch.distributed.barrier()
    return params_std


def write_dict_to_tensorboard(writer, wandb_writer, index, metrics, prefix=""):
    for key, value in metrics.items():
        key_to_write = prefix + key
        if isinstance(value, dict):
            write_dict_to_tensorboard(writer, wandb_writer, index, value, prefix=f"{key_to_write}/")
        else:
            writer.add_scalar(key_to_write, value, index)
            if wandb_writer is not None:
                wandb_writer.log({key_to_write: value}, index)


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    decoupled_learning_rate,
    iteration,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    params_std,
    num_zeros_in_grad,
    custom_metrics,
):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    all_logging_metrics = {}

    if custom_metrics is not None:
        all_logging_metrics["custom_metrics"] = custom_metrics

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float, device="cuda")) + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "all-grads-sync",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
    ]

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * get_num_microbatches()

    # Track app tag & app tag ID
    if one_logger:
        job_name = os.environ.get("SLURM_JOB_NAME", None)
        current_app_tag = f"{job_name}_{batch_size}_{args.world_size}"
        one_logger.log_app_tag(current_app_tag)

    total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration, normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({"samples vs steps": args.consumed_train_samples}, iteration)
        if args.log_learning_rate_to_tensorboard:
            writer.add_scalar("learning-rate", learning_rate, iteration)
            if args.decoupled_lr is not None:
                writer.add_scalar("decoupled-learning-rate", decoupled_learning_rate, iteration)
            writer.add_scalar("learning-rate vs samples", learning_rate, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"learning-rate": learning_rate}, iteration)
        if args.log_batch_size_to_tensorboard:
            writer.add_scalar("batch-size", batch_size, iteration)
            writer.add_scalar("batch-size vs samples", batch_size, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"batch-size": batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key, loss_dict[key], iteration)
            writer.add_scalar(key + " vs samples", loss_dict[key], args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar("loss-scale", loss_scale, iteration)
            writer.add_scalar("loss-scale vs samples", loss_scale, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"loss-scale": loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar("world-size", args.world_size, iteration)
            writer.add_scalar("world-size vs samples", args.world_size, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"world-size": args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar("grad-norm", grad_norm, iteration)
            writer.add_scalar("grad-norm vs samples", grad_norm, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"grad-norm": grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar("num-zeros", num_zeros_in_grad, iteration)
            writer.add_scalar("num-zeros vs samples", num_zeros_in_grad, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"num-zeros": num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar("params-norm", params_norm, iteration)
            writer.add_scalar("params-norm vs samples", params_norm, args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({"params-norm": params_norm}, iteration)
        if params_std is not None:
            write_dict_to_tensorboard(writer, wandb_writer, iteration, params_std, prefix="params-std/")
            write_dict_to_tensorboard(
                writer, None, args.consumed_train_samples, params_std, prefix="params-std vs samples/"
            )
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
        if custom_metrics is not None:
            write_dict_to_tensorboard(writer, wandb_writer, iteration, custom_metrics, prefix="custom_metrics/")
            write_dict_to_tensorboard(
                writer, None, args.consumed_train_samples, custom_metrics, prefix="custom_metrics vs samples/"
            )

    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(moe_loss_scale, iteration, writer, wandb_writer, total_loss_dict, args.moe_per_layer_logging)

    if iteration % args.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size
        )
        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar("iteration-time", elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({"iteration-time": elapsed_time_per_iteration}, iteration)

        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += " iteration {:8d}/{:8d} |".format(iteration, args.train_iters)
        log_string += " consumed samples: {:12d} |".format(args.consumed_train_samples)
        all_logging_metrics["consumed samples"] = args.consumed_train_samples
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(elapsed_time_per_iteration * 1000.0)
        all_logging_metrics["elapsed time per iteration (ms)"] = elapsed_time_per_iteration * 1000.0
        if args.log_throughput:
            log_string += f" throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |"
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar("throughput", throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({"throughput": throughput}, iteration)
            all_logging_metrics["TFLOPS"] = throughput
        assert learning_rate is not None
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += " learning rate: {:.6E} |".format(learning_rate)
        all_logging_metrics["learning rate"] = learning_rate
        if args.decoupled_lr is not None and (
            mpu.is_pipeline_first_stage(ignore_virtual=True) or mpu.is_pipeline_last_stage(ignore_virtual=True)
        ):
            assert decoupled_learning_rate is not None
            log_string += " decoupled learning rate: {:.6E} |".format(decoupled_learning_rate)
            all_logging_metrics["decoupled learning rate"] = decoupled_learning_rate
        else:
            assert decoupled_learning_rate is None
        log_string += " global batch size: {:5d} |".format(batch_size)
        all_logging_metrics["global batch size"] = batch_size
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                    all_logging_metrics[key] = avg
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device="cuda")
        log_string += " loss scale: {:.1f} |".format(loss_scale)
        all_logging_metrics["loss scale"] = loss_scale
        if grad_norm is not None:
            log_string += " grad norm: {:.3f} |".format(grad_norm)
            all_logging_metrics["grad norm"] = grad_norm
        if num_zeros_in_grad is not None:
            log_string += " num zeros: {:.1f} |".format(num_zeros_in_grad)
            all_logging_metrics["num zeros"] = num_zeros_in_grad
        if params_norm is not None:
            log_string += " params norm: {:.3f} |".format(params_norm)
            all_logging_metrics["params norm"] = params_norm
        if params_std is not None:
            params_std_to_str = "{"
            for key, value in params_std.items():
                params_std_to_str += f"{key} {value:.3f}, "
            params_std_to_str += "}"
            log_string += " params std: {} |".format(params_std_to_str)
            all_logging_metrics["params std"] = params_std
        log_string += " number of skipped iterations: {:3d} |".format(total_loss_dict[skipped_iters_key])
        all_logging_metrics["number of skipped iterations"] = total_loss_dict[skipped_iters_key]
        log_string += " number of nan iterations: {:3d} |".format(total_loss_dict[nan_iters_key])
        all_logging_metrics["number of nan iterations"] = total_loss_dict[nan_iters_key]
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag and learning_rate > 0.0:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory("(after {} iterations)".format(iteration))
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag, all_logging_metrics


def broadcast_spike_loss_ratio_in_pp_group(spike_loss_ratio: float):
    # Only last stage has actual spike_loss_ratio, that in other stages is None.
    # Thus, the broadcasting collective communication in pipeline group is necessary.
    if spike_loss_ratio is not None:
        src_spike_loss_tensor = torch.tensor(spike_loss_ratio, dtype=torch.float, device=torch.cuda.current_device())
        torch.distributed.broadcast(
            src_spike_loss_tensor,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
    else:
        dst_spike_loss_tensor = torch.tensor(1.0, dtype=torch.float, device=torch.cuda.current_device())
        torch.distributed.broadcast(
            dst_spike_loss_tensor,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        # means: if dst_spike_loss_tensor != 1.0
        if torch.abs(dst_spike_loss_tensor - 1.0) > 1e-6:
            spike_loss_ratio = dst_spike_loss_tensor.cpu().item()
    return spike_loss_ratio


def get_grad_norm_fp32(
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    norm_type: Union[int, float] = 2,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Norm parameters.
    norm_type = float(norm_type)
    total_norm = 0.0

    # Calculate norm.
    if norm_type == torch.inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device="cuda")
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group)
        total_norm = total_norm_cuda[0].item()

    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    amp_C.multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device="cuda")
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm**norm_type

        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group)
        total_norm = total_norm.item() ** (1.0 / norm_type)  # type: ignore[attr-defined]

    return total_norm


def get_main_grads_in_params(model_chunks: List[MegatronModule]):
    """
    Get main_grads that should be taken into account to scale the grad.
    Filter parameters based on:
        - grad should not be None.
        - parameter should not be shared (i.e., grads shouldn't be double counted while
        computing norms).
        - should not be a replica due to tensor model parallelism.
    """
    grads_for_scaling = []
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if hasattr(param, "main_grad") and param.main_grad is not None:
                grad = param.main_grad
            else:
                grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_scaling.append(grad)
    return grads_for_scaling


def scale_main_grad_for_spike_loss(
    model_chunks: List[MegatronModule], spike_loss_ratio: float, log_grad_diff_for_debug: bool = False
):
    """
    Scale main_grad of model params.
    """
    assert spike_loss_ratio is not None

    grads_for_scaling = get_main_grads_in_params(model_chunks)

    if log_grad_diff_for_debug:
        original_grad_norm = get_grad_norm_fp32(grads_for_scaling, model_parallel_group=mpu.get_model_parallel_group())

    # Scale grad for spike loss
    for grad in grads_for_scaling:
        grad *= spike_loss_ratio

    if log_grad_diff_for_debug:
        scaled_grad_norm = get_grad_norm_fp32(grads_for_scaling, model_parallel_group=mpu.get_model_parallel_group())
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] after scaling grad with ratio {spike_loss_ratio}, "
            f"grad norm {original_grad_norm} -> {scaled_grad_norm}"
        )


def get_grads_in_optimizer(optimizer: MegatronOptimizer):
    if isinstance(optimizer, (DistributedOptimizer, Float16OptimizerWithFloat16Params)):
        optimizer._copy_model_grads_to_main_grads()
        return optimizer.get_main_grads_for_grad_norm()
    elif isinstance(optimizer, ChainedOptimizer):
        grads_for_scaling = []
        for single_optimizer in optimizer.chained_optimizers:
            grads_for_scaling += get_grads_in_optimizer(single_optimizer)
        return grads_for_scaling
    else:
        raise ValueError(f"Unsupported optimizer type {type(optimizer)}")


def scale_grad_for_spike_loss(
    optimizer: MegatronOptimizer, spike_loss_ratio: float, log_grad_diff_for_debug: bool = False
):
    """
    Scale grad of optimizer.
    """
    assert spike_loss_ratio is not None

    grads_for_scaling = get_grads_in_optimizer(optimizer)

    if log_grad_diff_for_debug:
        original_grad_norm = get_grad_norm_fp32(grads_for_scaling, model_parallel_group=mpu.get_model_parallel_group())

    # Scale grad for spike loss
    for grad in grads_for_scaling:
        grad *= spike_loss_ratio

    if log_grad_diff_for_debug:
        scaled_grad_norm = get_grad_norm_fp32(grads_for_scaling, model_parallel_group=mpu.get_model_parallel_group())
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] after scaling grad with ratio {spike_loss_ratio}, "
            f"grad norm {original_grad_norm} -> {scaled_grad_norm}"
        )
