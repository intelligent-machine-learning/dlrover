# This implements a Dynamo compatible PiPPy compiler that transforms a graph module
# into a PiPPy distributed runtime
# The model construction is integrated with atorch distributed runtime
# FIXME Do more testing util integrate this with torch._dynamo.optimize
import math
from collections import OrderedDict

import torch
import torch.distributed.rpc as torch_rpc
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel

from atorch.amp.pipe_amp import get_pipe_amp_optimizer
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import (
    _DistributedContext,
    _prefix_pg_name,
    destroy_parallel_group,
    parallel_group,
    parallel_group_size,
    parallel_rank,
    rank,
)
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device
from atorch.utils.graph_transform_utils import map_aggregate

from .utils import (
    check_split_points,
    check_staged_model,
    compile_to_pipe,
    construct_output_chunk_spec,
    dp_pg_cb,
    get_number_of_params,
    hack_interpreter,
    hack_pippy_driver,
    prepare_args_kwargs,
    propagate_fake_split_gm,
)

try:
    from atorch.modules.distributed_modules.compilers.pipe_compiler.PipelineStage import (
        PipelineStage,
        PipelineStage1F1B,
    )
    from atorch.modules.distributed_modules.compilers.pipe_compiler.StageInterleaver import (
        InterleaverLRScheduler,
        InterleaverOptimizer,
        StageInterleaver,
    )

    pipeline_stage_imported = True
except ImportError:
    pipeline_stage_imported = False

try:
    from pippy import Pipe, PipelineDriver1F1B, PipelineDriverFillDrain
    from pippy.microbatch import Replicate, TensorChunkSpec, split_args_kwargs_into_chunks, sum_reducer
    from pippy.PipelineDriver import PipelineDriverInterleaved1F1B
except ImportError:
    Pipe, PipelineDriver1F1B, PipelineDriverFillDrain, PipelineDriverInterleaved1F1B = (
        None,
        None,
        None,
        None,
    )
    sum_reducer, TensorChunkSpec, Replicate, split_args_kwargs_into_chunks = None, None, None, None

_LRScheduler = None
try:
    from torch.optim.lr_scheduler import _LRScheduler  # type: ignore
except ImportError:
    _LRScheduler = object


driver_schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}

# Till 1F1B is supported
stage_schedules = {
    "FillDrain": PipelineStage,
    "1F1B": PipelineStage1F1B,
}


_COMPILED_DRIVER = None


class DummyOptim(torch.optim.Optimizer):
    def __init__(self):
        pass

    def zero_grad(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


class DummyLRScheduler(_LRScheduler):  # type: ignore
    def __init__(self):
        pass

    def step(self):
        pass


def _compiled_driver():
    global _COMPILED_DRIVER
    return _COMPILED_DRIVER


class SafeStage(torch.nn.Module):
    """A wrapper taht wraps around a StageInterleaver object, it extract necessary input,
    construct proper output that is compatible with normal work flow
    """

    def __init__(self, stage_interleaver, device, amp_config=None):
        super().__init__()
        self.stage_interleaver = stage_interleaver
        self.amp_config = amp_config
        self.device = device

    def forward(self, *args, **kwargs):
        def _lower_tensor_to_device(input_):
            if isinstance(input_, torch.Tensor):
                return (
                    input_.to(self.device)
                    if input_.device != torch.device("meta")
                    else torch.rand(input_.size(), device=self.device).to(dtype=input_.dtype)
                )
            else:
                return input_

        args = map_aggregate(args, _lower_tensor_to_device)
        kwargs = map_aggregate(kwargs, _lower_tensor_to_device)
        output = self.stage_interleaver(*args, **kwargs)
        return output

    # FIXME we should support param_func
    def instantiate_optimizer(self, optim_class, *args, **kwargs):
        def create_optimizer(pipe_stage):
            if self.amp_config is not None and self.amp_config["dtype"] != torch.bfloat16:
                pipe_opt = get_pipe_amp_optimizer(
                    pipe_stage.submod, pipe_stage.stage_index, optim_class, *args, **kwargs
                )
            else:
                pipe_opt = optim_class(pipe_stage.submod.parameters(), *args, **kwargs)
            return pipe_opt

        optimizers = map_aggregate(self.stage_interleaver.stages, create_optimizer)
        self.interleaved_optimizer = InterleaverOptimizer(optimizers)
        return self.interleaved_optimizer

    # FIXME add a lock to make sure optimizer initialized
    def instantiate_lr_scheduler(self, lr_sched_class, *args, **kwargs):
        def create_lr_scheduler(optimizer):
            return lr_sched_class(optimizer, *args, **kwargs)

        lr_schedulers = map_aggregate(self.interleaved_optimizer.optimizers, create_lr_scheduler)
        self.interleaved_lr_scheduler = InterleaverLRScheduler(lr_schedulers=lr_schedulers)
        return self.interleaved_lr_scheduler


class DeviceSafeDriver(torch.nn.Module):
    """A wrapper to guard the input in case of device mismatch.
    In dynamo compilation mode, wrapping the distributed pippy runtime
    with the DeviceSafeDriver makes it safe to compile a meta model.

    Args:
        pipe_drivers (torch.nn.Module): the distributed pippy runtime
        device (torch.device): the device that input_ should be on
        forward_keys (list): a list of keys giving the order of the input
    """

    def __init__(self, pipe_driver, device, pipe_ranks, forward_keys=None):
        super().__init__()
        self.pipe_driver = pipe_driver
        self.device = device
        self.forward_keys = forward_keys
        self.pipe_ranks = pipe_ranks
        logger.info(f"init device safe driver at {device}")

    def forward(self, *args, **kwargs):
        if rank() == self.pipe_ranks[0]:

            def _lower_tensor_to_device(input_):
                if isinstance(input_, torch.Tensor):
                    return (
                        input_.to(self.device)
                        if input_.device != torch.device("meta")
                        else torch.rand(input_.size(), device=self.device).to(dtype=input_.dtype)
                    )
                else:
                    return input_

            args = map_aggregate(args, _lower_tensor_to_device)
            kwargs = map_aggregate(kwargs, _lower_tensor_to_device)
            # adjust the order
            if self.forward_keys is not None and len(kwargs) > 0:
                kwargs = OrderedDict((k, kwargs[k]) for k in self.forward_keys if k in kwargs.keys())
            return self.pipe_driver(*args, **kwargs)
        else:
            # in case non pipe driver module, the worker process
            # should not be participating in the forward call in any occasions
            # so simply wait on the destroy_parallel_group call
            # and exit immediately upon completion of destruction
            logger.info(f"Rank {rank()} is not a pipe driver process, wait on parallel group destruction and exit")
            destroy_parallel_group()
            exit()

    # Simply make a call to the wrapped pipe driver
    def instantiate_optimizer(self, optim_class, *args, **kwargs):
        return (
            _compiled_driver().instantiate_optimizer(optim_class, *args, **kwargs)
            if rank() == self.pipe_ranks[0]
            else DummyOptim()
        )

    def instantiate_lr_scheduler(self, lr_sched_class, *args, **kwargs):
        return (
            _compiled_driver().instantiate_lr_scheduler(lr_sched_class, *args, **kwargs)
            if rank() == self.pipe_ranks[0]
            else DummyLRScheduler()
        )


def _compile_to_stage_mod(
    model_pipe,
    pipe_group,
    num_stages,
    device,
    chunks,
    pipe_schedule="1F1B",
    example_inputs=None,
    checkpoint=True,
    data_ranks=None,
    traced_forward_keys=None,
    amp_config=None,
    args_chunk_spec=None,
    kwargs_chunk_spec=None,
    output_chunk_spec=None,
    compiler_configs=dict(),
):
    # Example inputs are not micro-batched
    complete_args, complete_kwargs = prepare_args_kwargs(example_inputs, traced_forward_keys)
    # Use micro-batched args for shape inference
    complete_args, complete_kwargs = split_args_kwargs_into_chunks(
        complete_args,
        complete_kwargs,
        chunks,
        args_chunk_spec,
        kwargs_chunk_spec,
    )
    complete_args, complete_kwargs = complete_args[0], complete_kwargs[0]
    # install tensor meta onto split gm
    propagate_fake_split_gm(
        list(model_pipe.split_gm.graph.nodes), complete_args, complete_kwargs, compiler_configs=compiler_configs
    )
    pipe_rank = parallel_rank("pipe")
    pipe_size = parallel_group_size("pipe")

    stage_index = list(range(pipe_rank, num_stages, pipe_size))

    if pipe_schedule not in stage_schedules:
        pipe_schedule = "1F1B"

    def create_pipeline_stage(s_idx):
        pipe_stage = stage_schedules[pipe_schedule](
            pipe=model_pipe,
            stage_index=s_idx,
            nstages=num_stages,
            chunks=chunks,
            device=device,
            checkpoint=checkpoint,
            group=pipe_group,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec,
            forward_keys=traced_forward_keys,
        )
        # first materialize the module
        materialize_modules_to_device(pipe_stage.submod, device)
        pipe_stage.submod.to(device)

        # autocast the submod
        if amp_config is not None:
            pipe_stage.submod.forward = autocast(**amp_config)(pipe_stage.submod.forward)

        if data_ranks is not None:
            pipe_stage.submod = DistributedDataParallel(pipe_stage.submod, process_group=dp_pg_cb(pipe_rank))

        return pipe_stage

    pipe_stages = map_aggregate(stage_index, create_pipeline_stage)
    # In case not interleaved mode, Stage interleaver collapses into a wrapper for PipelineStage
    stage_interleaver = StageInterleaver(stages=pipe_stages)

    return SafeStage(stage_interleaver, device=device, amp_config=amp_config)


def _compile_to_driver(
    model_pipe,
    pipe_ranks,
    device,
    chunks,
    pipe_schedule="1F1B",
    use_c10d=False,
    checkpoint=True,
    data_ranks=None,
    traced_forward_keys=None,
    amp_config=None,
    args_chunk_spec=None,
    kwargs_chunk_spec=None,
    output_chunk_spec=None,
    compiler_configs=dict(),
):
    def pipe_defer_stage_init(pipe, device):
        def materialize_stage(target):
            logger.debug(f"materialization called on {rank()}")
            target_model = pipe.split_gm.get_submodule(target)
            materialize_modules_to_device(target_model, device)
            target_model.to(device)
            # Question: should we also hack the backward?
            if amp_config is not None:
                target_model.forward = autocast(**amp_config)(target_model.forward)
            return target_model

        with Pipe.stage_init_cv:
            setattr(Pipe, "materialize_stage", materialize_stage)
            Pipe.stage_init_cv.notify()

    setattr(Pipe, "defer_stage_init", pipe_defer_stage_init)

    # defer stage init and wait until driver rank finishes
    model_pipe.defer_stage_init(device)
    logger.debug(f"finish defer rewrite at {rank()}")

    # HACK the pipe stage executor to fix c10d bugs
    hack_pippy_driver()
    hack_interpreter(compiler_configs=compiler_configs)

    if rank() == pipe_ranks[0]:
        for i, sm in enumerate(model_pipe.split_gm.children()):
            logger.info(f"submod_{i} {round(get_number_of_params(sm) / 10 ** 9, 2)}B params")

    if rank() == pipe_ranks[0]:
        if pipe_schedule not in driver_schedules:
            logger.warning(f"pipe_schedule {pipe_schedule} is not supported, change to base version <FillDrain>")
            logger.warning(f"supported schedules: {driver_schedules.keys()}")
            pipe_schedule = "FillDrain"

        pipe_driver = driver_schedules[pipe_schedule](
            pipe=model_pipe,
            chunks=chunks,
            output_chunk_spec=output_chunk_spec,
            world_size=len(pipe_ranks),
            all_ranks=pipe_ranks,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            checkpoint=checkpoint,
            use_c10d=use_c10d,
        )
        if data_ranks is not None and len(data_ranks) > 1:
            pipe_driver.init_data_parallel(dp_group_size=len(data_ranks), dp_pg_cb=dp_pg_cb)

        global _COMPILED_DRIVER
        _COMPILED_DRIVER = pipe_driver
    else:
        pipe_driver = None

    compiled_pipe_driver = DeviceSafeDriver(pipe_driver, device, pipe_ranks, forward_keys=traced_forward_keys)

    return compiled_pipe_driver


def pippy_compiler(gm, data_configs=None, env_configs=None, compiler_configs=None):
    """The standalone compiler that can be used as a Dynamo backend (bugs could happen).
    In general this method compiles a torch.fx.GraphModule into a pipeline distributed runtime.

    Args:
        gm (torch.fx.GraphModule): the graph module to be transformed into pipe model,
            the correct split points must already be inserted
        data_configs (dict): a dict containing all input data related information
        env_configs (dict): a dict containing all environment related information, including the data
            and pipe ranks, groups
        compiler_configs (dict): a dict containing all info needed to correctly set up the pippy compiler
    """
    # parse arguments
    # For construct_output_chunk_spec and compile_stage, we need the complete (default/input) args/kwargs
    # to correctly do the shape inference.
    example_inputs = data_configs.get("example_inputs", None)
    input_batch_size = data_configs.get("input_batch_size", None)

    pipe_ranks = env_configs.get("pipe_ranks", None)
    data_ranks = env_configs.get("data_ranks", None)
    device = env_configs.get("device", None)

    model_input_format = compiler_configs.get("model_input_format", None)
    multi_use_param_spec = compiler_configs.get("multi_use_param_spec", None)
    output_loss_value_spec = compiler_configs.get("output_loss_value_spec", True)
    expected_num_stages = compiler_configs.get("expected_num_stages", None)
    compile_with_dynamo = compiler_configs.get("compile_with_dynamo", False)
    export_mode = compiler_configs.get("export_mode", False)
    chunks = compiler_configs.get("chunks", None)
    if chunks > 1:
        logger.info(f"Note that the loss must be divided by chunks {chunks} to reflect the actual average loss")
    pipe_schedule = compiler_configs.get("pipe_schedule", "1F1B")
    # default to checkpointing to save memory?
    checkpoint = compiler_configs.get("checkpoint", False)
    checkpoint_keys = compiler_configs.get("checkpoint_keys", dict())
    use_c10d = compiler_configs.get("use_c10d", False)
    if use_c10d:
        logger.info("Note that only last pipe ranks returns real loss")
    train_mode = compiler_configs.get("train_mode", True)
    amp_config = compiler_configs.get("amp_config", None)
    # Safety step that cleans up the amp_config
    amp_config = (
        {k: amp_config[k] for k in ["enabled", "dtype", "cache_enabled"] if k in amp_config}
        if amp_config is not None
        else None
    )

    model_pipe, traced_forward_keys = compile_to_pipe(
        gm, pipe_ranks, compile_with_dynamo, multi_use_param_spec, output_loss_value_spec, device
    )
    complete_args, complete_kwargs = prepare_args_kwargs(example_inputs, traced_forward_keys)

    if train_mode:
        # output_chunk_spec is a reducer as we are using loss wrapper
        output_chunk_spec = sum_reducer
    else:
        # For inference mode, we do a fake shape prop to infer the output_chunk_spec
        output_chunk_spec = construct_output_chunk_spec(complete_args, complete_kwargs)

    num_split_points = check_split_points(gm)
    if num_split_points >= len(pipe_ranks) and use_c10d:
        logger.info(f"{num_split_points + 1} stages, only {len(pipe_ranks)} pipe ranks, use interleaved schedule")
        pipe_schedule = "Interleaved1F1B"

    if checkpoint_keys is not None:
        check_staged_model(model_pipe.split_gm, checkpoint_keys)
    num_stages = len(list(model_pipe.split_gm.children()))
    expected_num_stages = expected_num_stages or len(pipe_ranks)
    if num_stages != expected_num_stages:
        logger.info(f"Model is split into {num_stages} instead of {expected_num_stages} stages")

    # creating extra data groups for interleaved schedule
    if num_stages > len(pipe_ranks) and data_ranks is not None and len(data_ranks) > 1:
        # wait here so that all init data parallel can finish
        torch_rpc.api._wait_all_workers()
        num_indices = math.ceil(num_stages / len(pipe_ranks))
        data_groups_and_ranks = _DistributedContext.PARALLEL_GROUPS_AND_RANKS[_prefix_pg_name("data")]

        for (_, ranks) in data_groups_and_ranks:
            for idx in range(1, num_indices):
                new_data_group = torch.distributed.new_group(ranks)
                if rank() in ranks:
                    _DistributedContext.PARALLEL_GROUP[_prefix_pg_name(f"data_{idx}")] = new_data_group
                    _DistributedContext.PARALLEL_RANK[_prefix_pg_name(f"data_{idx}")] = ranks.index(rank())
                    _DistributedContext.PARALLEL_GROUP_SIZE[_prefix_pg_name(f"data_{idx}")] = len(data_ranks)
        # wait here so that all init data parallel can finish
        torch_rpc.api._wait_all_workers()

    # generate chunk specs
    # assuming the first dimension is always the batch dimension
    def chunk_spec_gen(input_):
        if isinstance(input_, torch.Tensor) and len(list(input_.size())) > 0:
            if input_batch_size is not None:
                # use batch_size to further verify the chunk dimension
                input_size = list(input_.size())
                if input_batch_size in input_size:
                    # use the first matching dimension as the batch dimension
                    return TensorChunkSpec(input_size.index(input_batch_size))
                else:
                    # in this case assume not shardable
                    return Replicate()
            else:
                return None
        else:
            return Replicate()

    input_chunk_spec = map_aggregate(example_inputs, chunk_spec_gen)
    args_chunk_spec = ()
    kwargs_chunk_spec = {}

    keyed_spec = None
    if compile_with_dynamo and not export_mode:
        # in the case of dynamo compile mode, placeholders will be limited to input's keys only
        # example_inputs will be turned into a list, matching the placeholder
        keyed_spec = dict(zip(traced_forward_keys.keys(), input_chunk_spec))
    else:
        # in fx mode and export mode, example_inputs are the name<->value paired dict
        # map_aggregate on this naturally is kwargs_chunk_spec
        keyed_spec = input_chunk_spec

    if model_input_format is None:
        args_chunk_spec = (next(iter(keyed_spec.values())),)
    elif model_input_format == "unpack_sequence":
        args_chunk_spec = tuple(keyed_spec.values())
    else:
        kwargs_chunk_spec = keyed_spec

    if not use_c10d or not pipeline_stage_imported:
        return _compile_to_driver(
            model_pipe=model_pipe,
            pipe_ranks=pipe_ranks,
            device=device,
            chunks=chunks,
            pipe_schedule=pipe_schedule,
            use_c10d=use_c10d,
            checkpoint=checkpoint,
            data_ranks=data_ranks,
            traced_forward_keys=traced_forward_keys.keys(),
            amp_config=amp_config,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec,
            compiler_configs=compiler_configs,
        )
    else:
        return _compile_to_stage_mod(
            model_pipe=model_pipe,
            pipe_group=parallel_group("pipe"),
            num_stages=num_stages,
            device=device,
            chunks=chunks,
            pipe_schedule=pipe_schedule,
            example_inputs=example_inputs,
            checkpoint=checkpoint,
            data_ranks=data_ranks,
            traced_forward_keys=traced_forward_keys,
            amp_config=amp_config,
            # FIXME cannot specify the full spec, so we let PiPPy generate it on the fly
            # args_chunk_spec=args_chunk_spec,
            # kwargs_chunk_spec=kwargs_chunk_spec,
            output_chunk_spec=output_chunk_spec,
            compiler_configs=compiler_configs,
        )
