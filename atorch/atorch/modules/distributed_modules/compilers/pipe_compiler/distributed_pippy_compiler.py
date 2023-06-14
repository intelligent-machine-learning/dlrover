# This implements a Dynamo compatible PiPPy compiler that transforms a graph module
# into a PiPPy distributed runtime
# The model construction is integrated with atorch distributed runtime
# FIXME Do more testing util integrate this with torch._dynamo.optimize
import contextlib
import inspect
import time
from collections import OrderedDict
from typing import List

import torch

try:
    from torch._subclasses.fake_tensor import FakeTensorMode
except ImportError:
    FakeTensorMode = object

try:
    import pippy
    from pippy import Pipe, PipelineDriver1F1B, PipelineDriverFillDrain
    from pippy.fx.node import map_aggregate as pippy_map_aggregate
    from pippy.fx.passes.shape_prop import ShapeProp, TensorMetadata
    from pippy.IR import pipe_split
    from pippy.microbatch import Replicate, TensorChunkSpec, sum_reducer
    from pippy.PipelineDriver import (
        PipelineDriverInterleaved1F1B,
        PipeStageExecutor,
        RefcountedFuture,
        RemoteInterpreter,
        SchedState,
        ValueReference,
        WorkItem,
        event_id,
        event_name,
    )
except ImportError:
    pippy = None
    Pipe, PipelineDriver1F1B, PipelineDriverFillDrain, PipelineDriverInterleaved1F1B = None, None, None, None
    sum_reducer, TensorChunkSpec, Replicate, pipe_split = None, None, None, None
    RemoteInterpreter, PipeStageExecutor = None, None
    event_name, ValueReference, SchedState, RefcountedFuture, event_id, WorkItem = None, None, None, None, None, None
    from torch.fx.node import map_aggregate as pippy_map_aggregate
    from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    LRScheduler = object

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import destroy_parallel_group, local_rank, parallel_group, rank
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device
from atorch.utils.graph_transform_utils import map_aggregate

schedules = {
    "FillDrain": PipelineDriverFillDrain,
    "1F1B": PipelineDriver1F1B,
    "Interleaved1F1B": PipelineDriverInterleaved1F1B,
}


_COMPILED_DRIVER = None


@contextlib.contextmanager
def retain_graph_context():
    def new_backward(
        tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None, inputs=None
    ):
        return old_backward(tensors, grad_tensors, True, create_graph, grad_variables, inputs)

    old_backward = torch.autograd.backward
    torch.autograd.backward = new_backward
    try:
        yield
    finally:
        torch.autograd.backward = old_backward


class RetainGradShapeProp(ShapeProp):
    def run_node(self, n):
        # Since in Pipe graph, backward are manually called,
        # we need to manually retain the intermediate tensors to properly propagate shapes
        result = super().run_node(n)

        def retain_grad_func(input_):
            if isinstance(input_, torch.Tensor) and input_.requires_grad:
                input_.retain_grad()
            return input_

        # In fake tensor shape prop, output loss (a scaler) will be of size torch.Size([]),
        # this is not compatible with actual sizing in actual pipeline training, fix it here
        def fix_singleton_size(input_):
            if isinstance(input_, TensorMetadata) and input_.shape == torch.Size([]):
                input_ = TensorMetadata(
                    shape=torch.Size([1]),
                    dtype=input_.dtype,
                    requires_grad=input_.requires_grad,
                    stride=input_.stride,
                    memory_format=input_.memory_format,
                    is_quantized=input_.is_quantized,
                    qparams=input_.qparams,
                )
            return input_

        result = map_aggregate(result, retain_grad_func)
        if "tensor_meta" in n.meta:
            tm = n.meta["tensor_meta"]
            n.meta["tensor_meta"] = pippy_map_aggregate(
                tm, fix_singleton_size, lambda a: not isinstance(a, TensorMetadata)
            )
        return result


class DummyOptim(torch.optim.Optimizer):
    def __init__(self):
        pass

    def zero_grad(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass


class DummyLRScheduler(LRScheduler):
    def __init__(self):
        pass

    def step(self):
        pass


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _compiled_driver():
    global _COMPILED_DRIVER
    return _COMPILED_DRIVER


def dp_pg_cb(stage_id):
    return parallel_group("data")


# FIXME Is this safe is the original model is TP?
def rewrite_to_device(graph, device):
    def rewrite_device(input_):
        if isinstance(input_, torch.device):
            return device
        else:
            return input_

    for node in graph.nodes:
        new_args = map_aggregate(node.args, rewrite_device)
        new_kwargs = map_aggregate(node.kwargs, rewrite_device)
        node.args = new_args
        node.kwargs = new_kwargs


def check_staged_model(staged_model, checkpoint_keys):
    """A safe guard for pippy split model. Upon splitting model,
    Pippy will flatten and attach some model parameters to the top-level
    stage model (for instance, multi-used parameters). If the model is offloaded to meta,
    we need to assign the checkpoint name back to these flatten params.
    """
    # check for moved model parameters
    flatten_names = {name.replace(".", "_"): name for name in checkpoint_keys.keys()}

    def _fix_checkpoint_name(stage_mod, param_name, param):
        if (
            not hasattr(param, "checkpoint_name")
            and isinstance(param, torch.Tensor)
            and param.device == torch.device("meta")
        ):
            if param_name.startswith("moved_"):
                flatten_name = param_name[6:].replace(".", "_")
            else:
                flatten_name = param_name.replace(".", "_")
            if flatten_name.startswith("self_"):
                flatten_name = flatten_name[5:]
            orig_name = flatten_names[flatten_name]
            checkpoint_name = checkpoint_keys[orig_name]
            if orig_name in checkpoint_keys:
                atoms = param_name.split(".")
                mod_itr = stage_mod
                for atom in atoms[:-1]:
                    mod_itr = getattr(mod_itr, atom)
                param_val = getattr(mod_itr, atoms[-1])
                setattr(param_val, "checkpoint_name", checkpoint_name)
            else:
                logger.info(f"param: {orig_name} is not checkpointed")

    # replicated params will be registered at stage level
    for stage_mod in staged_model.children():
        for param_name, param in stage_mod.named_parameters():
            _fix_checkpoint_name(stage_mod, param_name, param)

        for buffer_name, stage_buffer in stage_mod.named_buffers():
            _fix_checkpoint_name(stage_mod, buffer_name, stage_buffer)


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


def _check_split_points(gm):
    graph = gm.graph
    num_split_points = 0
    for node in graph.nodes:
        if (node.op, node.target) == ("call_function", pipe_split):
            num_split_points += 1
    return num_split_points


def _compile_to_pipe(
    gm, pipe_ranks, compile_with_dynamo=False, multi_use_param_spec=None, output_loss_value_spec=True, device=None
):
    graph = gm.graph
    output_node = [n for n in graph.nodes if n.op == "output"][0]
    if isinstance(output_node.args[0], (tuple, list)) and output_loss_value_spec is True:
        output_loss_value_spec = (True,)

    # Assume that the graph is generated by model_context.capture_compute_graph
    # and graph rewrite is needed
    if compile_with_dynamo:
        device = device or torch.device("cuda:{}".format(local_rank()))
        rewrite_to_device(graph, device)

    if rank() == pipe_ranks[0] and logger.root.level > 30:
        logger.info("corrected graph")
        graph.print_tabular()

    # Turn the graph module into a pippy compatible one
    if isinstance(graph, torch.fx.Graph):
        # HACK to convert torch.fx.Graph to pippy.fx.Graph
        new_graph = pippy.fx.Graph()
        val_map = {}
        out = new_graph.graph_copy(graph, val_map, False)
        new_graph.output(out)

        # `pippy.fx.map_arg` doesn't work on torch.fx.Node instances;
        # do it here
        def remap_vals(n):
            return val_map[n]

        for node in new_graph.nodes:
            node.args = torch.fx.map_arg(node.args, remap_vals)
            node.kwargs = torch.fx.map_arg(node.kwargs, remap_vals)
        graph = new_graph

    # compile the graph into a PiPPy GraphModule
    traced = pippy.fx.GraphModule(gm, new_graph)
    traced_forward_keys = inspect.signature(traced.forward).parameters.keys()

    model_pipe = Pipe._from_traced(
        gm,
        traced,
        multi_use_param_spec,
        output_loss_value_spec=output_loss_value_spec,
    )
    return model_pipe, traced_forward_keys


def pippy_compiler(gm, data_configs=None, env_configs=None, compiler_configs=None):
    """The standalone compiler that can be used as a Dynamo backend (bugs could happen).
    In general this method compiles a torch.fx.GraphModule into a pipeline distributed runtime.

    Args:
        gm (torch.fx.GraphModule): the graph module to be transformed into pipe model,
            the correct split points must already be correctly inserted
        data_configs (dict): a dict containing all input data related information
        env_configs (dict): a dict containing all environment related information, including the data
            and pipe ranks, groups
        compiler_configs (dict): a dict containing all info needed to correctly set up the pippy compiler
    """
    # parse arguments
    example_inputs = data_configs.get("example_inputs", None)
    input_batch_size = data_configs.get("input_batch_size", None)

    pipe_ranks = env_configs.get("pipe_ranks", None)
    data_ranks = env_configs.get("data_ranks", None)
    device = env_configs.get("device", None)

    multi_use_param_spec = compiler_configs.get("multi_use_param_spec", None)
    output_loss_value_spec = compiler_configs.get("output_loss_value_spec", True)
    expected_num_stages = compiler_configs.get("expected_num_stages", None)
    compile_with_dynamo = compiler_configs.get("compile_with_dynamo", False)
    export_mode = compiler_configs.get("export_mode", False)
    chunks = compiler_configs.get("chunks", None)
    pipe_schedule = compiler_configs.get("pipe_schedule", "1F1B")
    # default to checkpointing to save memory?
    checkpoint = compiler_configs.get("checkpoint", False)
    checkpoint_keys = compiler_configs.get("checkpoint_keys", dict())
    use_c10d = compiler_configs.get("use_c10d", False)
    train_mode = compiler_configs.get("train_mode", True)
    dynamic_shape = compiler_configs.get("dynamic_shape", False)

    num_split_points = _check_split_points(gm)
    if num_split_points >= len(pipe_ranks):
        logger.info(f"{num_split_points + 1} stages, only {len(pipe_ranks)} pipe ranks, use interleaved schedule")
        pipe_schedule = "Interleaved1F1B"

    model_pipe, traced_forward_keys = _compile_to_pipe(
        gm, pipe_ranks, compile_with_dynamo, multi_use_param_spec, output_loss_value_spec, device
    )

    if checkpoint_keys is not None:
        check_staged_model(model_pipe.split_gm, checkpoint_keys)
    num_stages = len(list(model_pipe.split_gm.children()))
    expected_num_stages = expected_num_stages or len(pipe_ranks)
    if num_stages != expected_num_stages:
        logger.info(f"Model is split into {num_stages} instead of {expected_num_stages} stages")

    # defer init
    # new version of pippy relies on thread lock to properly
    # defer stage init

    def pipe_defer_stage_init(pipe, device):
        def materialize_stage(target):
            logger.debug(f"materialization called on {rank()}")
            target_model = pipe.split_gm.get_submodule(target)
            materialize_modules_to_device(target_model, device)
            target_model.to(device)
            return target_model

        with Pipe.stage_init_cv:
            setattr(Pipe, "materialize_stage", materialize_stage)
            Pipe.stage_init_cv.notify()

    setattr(Pipe, "defer_stage_init", pipe_defer_stage_init)

    # defer stage init and wait until driver rank finishes
    model_pipe.defer_stage_init(device)
    logger.debug(f"finish defer rewrite at {rank()}")

    # HACK the pipe stage executor to fix c10d bugs
    if use_c10d:

        def safe_invoke(
            executor,
            output_unique_key,
            phase,
            args,
            kwargs,
            cur_microbatch,
            debug_str,
            output_refcount,
            batch_id,
            num_microbatches,
        ):
            start_ts = time.time()
            target_name = event_name(phase, executor.stage_id, cur_microbatch)
            target_id = event_id(phase, executor.stage_id, cur_microbatch, batch_id)
            name = f"R{target_name}"
            id = f"R{target_id}"
            if executor._record_mem_dumps:
                executor._record_dumps_on_all_peer_executors(f"M{id}_invoke", start_ts)

            logger.debug(f"[{executor.stage_id}][{cur_microbatch}] Received invoke call for {debug_str}")
            value_ref_args = []

            def extract_value_ref_args(arg):
                if isinstance(arg, ValueReference) and arg.unique_key != "noop":
                    value_ref_args.append(arg)

            pippy.fx.node.map_aggregate(args, extract_value_ref_args)
            pippy.fx.node.map_aggregate(kwargs, extract_value_ref_args)

            logger.debug(
                f"[{executor.stage_id}][{cur_microbatch}] Invoke "
                f"call found {len(value_ref_args)} ValueReference arguments"
            )
            future: torch.futures.Future = executor.create_future()

            work_item = WorkItem(
                stage_id=executor.stage_id,
                phase=phase,
                args=args,
                kwargs=kwargs,
                future=future,
                microbatch_id=cur_microbatch,
                blocked_args_count=len(value_ref_args),
                ready_args={},
                batch_id=batch_id,
                num_microbatches=num_microbatches,
                debug_str=debug_str,
            )
            logger.debug(
                f"[{executor.stage_id}][{cur_microbatch}] Invoke "
                f"instantiated WorkItem {work_item} with key {output_unique_key}"
            )
            if len(value_ref_args) == 0:
                work_item.state = SchedState.READY
                logger.debug(
                    f"[{executor.stage_id}][{cur_microbatch}] No RRef arguments. "
                    f"Scheduling directly as READY workitem"
                )
                executor.rank_worker.enqueue_ready_runlist(output_unique_key, work_item)
            else:
                logger.debug(f"[{executor.stage_id}][{cur_microbatch}] Scheduling WorkItem as WAITING workitem")
                work_item.state = SchedState.WAITING
                executor.rank_worker.enqueue_waiting_runlist(output_unique_key, work_item)

            callee_stage_dict = {}
            for arg_idx, value_ref_arg in enumerate(value_ref_args):
                # HACK, check here to make don't send anything to a rank itself through torch.distributed.isend
                if "tensor_meta" in value_ref_arg.meta and value_ref_arg.stage_id != executor.stage_id:
                    callee_stage = value_ref_arg.stage_id
                    batch_refs = callee_stage_dict.setdefault(callee_stage, {})
                    batch_refs[arg_idx] = value_ref_arg
                else:
                    # For non-tensor (e.g. a value or a size vector), we use RPC to spawn asynchronous data transfer
                    logger.debug(
                        f"[{executor.stage_id}][{cur_microbatch}] Launching RPC data transfer for "
                        f"ValueReference {arg_idx} {value_ref_arg}"
                    )
                    executor.async_transfer(cur_microbatch, value_ref_arg, arg_idx, output_unique_key)

            with executor.callee_send_tag_lock:
                for callee_stage, batch_refs in callee_stage_dict.items():
                    value_ref_executor_rref = executor.peer_executors[callee_stage]
                    tag = executor.callee_send_tag.setdefault(callee_stage, 0)
                    executor.callee_send_tag[callee_stage] += 1
                    value_ref_executor_rref.rpc_async().batch_send(
                        executor.stage_id,
                        output_unique_key,
                        cur_microbatch,
                        batch_refs,
                        tag,
                    )
                    executor.batch_recv(
                        cur_microbatch,
                        output_unique_key,
                        callee_stage,
                        batch_refs,
                        tag,
                    )

            with executor.value_store_cv:
                assert output_unique_key not in executor.value_store, (
                    f"[{executor.stage_id}] Output key {output_unique_key} "
                    f"already exists or is not consumed from previous batch"
                )
                executor.value_store[output_unique_key] = RefcountedFuture(future, output_refcount)
                executor.value_store_cv.notify_all()

            finish_ts = time.time()
            executor.record_event(
                rank=executor.rank_worker.rank,
                start_ts=start_ts,
                finish_ts=finish_ts,
                id=id,
                name=name,
                type="received",
                mbid=cur_microbatch,
            )
            executor.record_event_dependency(from_id=name, to_id=target_name, type="waiting")

            return ValueReference(executor.stage_id, output_unique_key)

        setattr(PipeStageExecutor, "invoke", safe_invoke)

        def safe_batch_recv(executor, microbatch, runlist_key, callee_stage, batch_refs, tag):
            logger.debug(
                f"[{executor.stage_id}][{microbatch}] Receiving batch {tag} of {len(batch_refs)} values "
                f"for runlist item {runlist_key} from stage {callee_stage}"
            )
            futures = []

            for arg_idx, value_ref_arg in batch_refs.items():
                tm = value_ref_arg.meta["tensor_meta"]
                # HACK here: must set requires_grad
                recv_buff = torch.empty(
                    tm.shape, dtype=tm.dtype, device=executor.device, requires_grad=tm.requires_grad
                )

                if torch.distributed.get_backend() == "gloo":
                    fut: torch.futures.Future = executor.create_future()
                    torch.distributed.recv(recv_buff, callee_stage, tag=tag)
                    fut.set_result(recv_buff)
                else:
                    work = torch.distributed.irecv(recv_buff, callee_stage, tag=tag)
                    fut = work.get_future()  # type: ignore[attr-defined]

                def bottom_half(fut):
                    logger.debug(
                        f"[{executor.stage_id}][{microbatch}] Completing transfer of value {value_ref_arg} "
                        f"for runlist item {runlist_key} arg_idx {arg_idx}"
                    )
                    value = fut.value()
                    if isinstance(value, List):
                        value = value[0]
                    executor.rank_worker.update_run_list(runlist_key, arg_idx, value)

                futures.append(fut.then(bottom_half))

            return futures

        setattr(PipeStageExecutor, "batch_recv", safe_batch_recv)

    if rank() == pipe_ranks[0]:
        for i, sm in enumerate(model_pipe.split_gm.children()):
            logger.info(f"submod_{i} {round(get_number_of_params(sm) / 10 ** 9, 2)}B params")

    if rank() == pipe_ranks[0]:
        # prepare chunk spec for input
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
        # output_chunk_spec is a reducer as we are using loss wrapper
        output_chunk_spec = sum_reducer if train_mode else None
        if compile_with_dynamo and not export_mode:
            # in the case of dynamo compile mode, placeholders will be limited to input's keys only
            # example_inputs will be turned into a list, matching the placeholder
            args_chunk_spec = input_chunk_spec
        else:
            # in fx mode and export mode, example_inputs are the name<->value paired dict
            # map_aggregate on this naturally is kwargs_chunk_spec
            kwargs_chunk_spec = input_chunk_spec

        if rank() == pipe_ranks[0]:
            logger.debug(f"input_chunk_spec: {input_chunk_spec}")
        if pipe_schedule not in schedules:
            logger.warning(f"pipe_schedule {pipe_schedule} is not supported, change to base version <FillDrain>")
            logger.warning(f"supported schedules: {schedules.keys()}")
            pipe_schedule = "FillDrain"

        if use_c10d:
            counter = AutoAccelerateContext.counter
            if not hasattr(AutoAccelerateContext, "fake_prop_done"):
                AutoAccelerateContext.add_ac_attr("fake_prop_done", {counter: False})
            else:
                AutoAccelerateContext.fake_prop_done[counter] = False

            def install_tensor_meta(fake_node_list, node_list):
                for fake_node, node in zip(fake_node_list, node_list):
                    if "tensor_meta" in fake_node.meta:
                        node.meta["tensor_meta"] = fake_node.meta["tensor_meta"]

            def _propagate_fake_split_gm(args, kwargs):
                counter = AutoAccelerateContext.counter
                fake_split_gm = AutoAccelerateContext.fake_split_gm[counter]
                fake_mode = fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
                complete_args = list(args) + list(kwargs.values())
                fake_args = map_aggregate(
                    complete_args,
                    lambda a: fake_mode.from_tensor(a.to("meta")) if isinstance(a, torch.Tensor) else None,
                )

                # Each stage_backward node will make a call to autograd.backward
                # We have to default the retain_graph to True to execute the whole graph
                with retain_graph_context():
                    sp = RetainGradShapeProp(fake_split_gm)
                    sp.propagate(*fake_args)
                logger.info("finish propagating fake split gm")

            def _propagate_shape_fake(interp, args, kwargs):
                logger.info("Propagating shape across fake split GraphModule")
                counter = AutoAccelerateContext.counter
                fake_split_gm = AutoAccelerateContext.fake_split_gm[counter]
                if dynamic_shape or not AutoAccelerateContext.fake_prop_done[counter]:
                    AutoAccelerateContext.fake_prop_done[counter] = True
                    _propagate_fake_split_gm(args, kwargs)

                install_tensor_meta(list(fake_split_gm.graph.nodes), interp.node_list)

            setattr(RemoteInterpreter, "propagate_shape", _propagate_shape_fake)
            # FIXME consider enable installation of tensor_meta onto call_function
            # this would make the backward pass also c10d

        pipe_driver = schedules[pipe_schedule](
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
