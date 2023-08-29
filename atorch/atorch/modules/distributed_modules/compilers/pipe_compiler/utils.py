# Here implements functionalities that bypasses the defaults of PiPPy
import contextlib
import copy
import inspect
import time
from typing import List

import torch

try:
    from torch._subclasses.fake_tensor import FakeTensorMode
except ImportError:
    FakeTensorMode = object

try:
    import pippy
    from pippy import Pipe
    from pippy.fx.interpreter import Interpreter
    from pippy.fx.node import map_aggregate as pippy_map_aggregate
    from pippy.fx.passes.shape_prop import ShapeProp, TensorMetadata
    from pippy.IR import pipe_split
    from pippy.microbatch import TensorChunkSpec
    from pippy.PipelineDriver import (
        DEBUG,
        PipelineDriverBase,
        PipeStageExecutor,
        RefcountedFuture,
        RemoteInterpreter,
        SchedState,
        ValueReference,
        WorkItem,
        _wait_for_all,
        event_id,
        event_name,
    )
    from torch.fx.passes.shape_prop import TensorMetadata as FXTensorMetadata
except ImportError:
    pippy, Pipe, PipelineDriverBase, pipe_split = None, None, None, None
    TensorChunkSpec, RemoteInterpreter, PipeStageExecutor = None, None, None
    _wait_for_all, event_name, ValueReference, SchedState, RefcountedFuture, event_id, WorkItem = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    from torch.fx.interpreter import Interpreter
    from torch.fx.node import map_aggregate as pippy_map_aggregate
    from torch.fx.passes.shape_prop import ShapeProp, TensorMetadata

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import (
    _DistributedContext,
    _prefix_pg_name,
    local_rank,
    parallel_group,
    parallel_group_and_ranks,
    parallel_group_size,
    rank,
)
from atorch.utils.graph_transform_utils import map_aggregate
from atorch.utils.version import torch_version


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

        result = map_aggregate(result, retain_grad_func)
        return result


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dp_pg_cb(stage_id):
    pipe_size = parallel_group_size("pipe")
    idx = stage_id // pipe_size
    data_group = (
        parallel_group("data") if idx == 0 else _DistributedContext.PARALLEL_GROUP[_prefix_pg_name(f"data_{idx}")]
    )
    return data_group


# FIXME Is this safe if the original model is TP?
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


def hack_pippy_driver():
    def send_recv(executor, caller_stage, runlist_key, microbatch, batch_refs, tag):
        with executor.caller_recv_tag_cv:
            executor.caller_recv_tag.setdefault(caller_stage, 0)
            while executor.caller_recv_tag[caller_stage] < tag:
                executor.caller_recv_tag_cv.wait()

        logger.debug(
            f"[{executor.stage_id}][{microbatch}] Sending batch {tag} of "
            f"{len(batch_refs)} values initiated by stage {caller_stage} for {runlist_key}"
        )

        for arg_idx, value_ref_arg in batch_refs.items():
            with executor.value_store_cv:
                # Waiting for the indexed future for this arg to be created
                while value_ref_arg.unique_key not in executor.value_store:
                    executor.value_store_cv.wait()
                # Now the indexed future is created
                refcounted_future = executor.value_store[value_ref_arg.unique_key]

            value = refcounted_future.future.wait()

            with executor.value_store_lock:
                if refcounted_future.release():
                    executor.value_store.pop(value_ref_arg.unique_key)

            # minic the recv behaviour here
            if isinstance(value, List):
                value = value[0]
            executor.rank_worker.update_run_list(runlist_key, arg_idx, value)

        # Notify next send that's potentially waiting
        with executor.caller_recv_tag_cv:
            executor.caller_recv_tag[caller_stage] += 1
            executor.caller_recv_tag_cv.notify_all()

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
                f"[{executor.stage_id}][{cur_microbatch}] No RRef arguments. " f"Scheduling directly as READY workitem"
            )
            executor.rank_worker.enqueue_ready_runlist(output_unique_key, work_item)
        else:
            logger.debug(f"[{executor.stage_id}][{cur_microbatch}] Scheduling WorkItem as WAITING workitem")
            work_item.state = SchedState.WAITING
            executor.rank_worker.enqueue_waiting_runlist(output_unique_key, work_item)

        callee_stage_dict = {}
        for arg_idx, value_ref_arg in enumerate(value_ref_args):
            # HACK, check here to make don't send anything to a rank itself through torch.distributed.isend
            if "tensor_meta" in value_ref_arg.meta:
                callee_stage = value_ref_arg.stage_id
                batch_refs = callee_stage_dict.setdefault(callee_stage, {})
                batch_refs[arg_idx] = value_ref_arg
            else:
                # For non-tensor (e.g. a value or a size vector), we use RPC to spawn asynchronous data transfer
                logger.debug(
                    f"[{executor.stage_id}][{cur_microbatch}] Launching RPC data transfer for "
                    f"ValueReference {arg_idx} {value_ref_arg}"
                )
                callee_stage = value_ref_arg.stage_id
                if callee_stage == executor.stage_id:
                    logger.debug(f"Bypassing RPC data transfer for local data ref: {value_ref_arg}")
                    value = executor.get_value(callee_stage, output_unique_key, cur_microbatch, value_ref_arg)
                    executor.rank_worker.update_run_list(output_unique_key, arg_idx, value)
                else:
                    executor.async_transfer(cur_microbatch, value_ref_arg, arg_idx, output_unique_key)

        with executor.callee_send_tag_lock:
            for callee_stage, batch_refs in callee_stage_dict.items():
                tag = executor.callee_send_tag.setdefault(callee_stage, 0)
                if callee_stage != executor.stage_id:
                    value_ref_executor_rref = executor.peer_executors[callee_stage]
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
                else:
                    logger.debug(f"Bypassing c10d data transfer for local data ref: {value_ref_arg}")
                    send_recv(executor, executor.stage_id, output_unique_key, cur_microbatch, batch_refs, tag)

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
            recv_buff = torch.empty(tm.shape, dtype=tm.dtype, device=executor.device, requires_grad=tm.requires_grad)

            # HACK irecv causes the pipeline training to hang
            fut = executor.create_future()
            torch.distributed.recv(recv_buff, callee_stage, tag=tag)
            fut.set_result(recv_buff)

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

    def safe_batch_send(
        executor,
        caller_stage,
        runlist_key,
        microbatch,
        batch_refs,
        tag,
    ):
        # Wait till this batch's turn to send
        with executor.caller_recv_tag_cv:
            executor.caller_recv_tag.setdefault(caller_stage, 0)
            while executor.caller_recv_tag[caller_stage] < tag:
                executor.caller_recv_tag_cv.wait()

        logger.debug(
            f"[{executor.stage_id}][{microbatch}] Sending batch {tag} of "
            f"{len(batch_refs)} values initiated by stage {caller_stage} for {runlist_key}"
        )

        for _, value_ref_arg in batch_refs.items():
            with executor.value_store_cv:
                # Waiting for the indexed future for this arg to be created
                while value_ref_arg.unique_key not in executor.value_store:
                    executor.value_store_cv.wait()
                # Now the indexed future is created
                refcounted_future = executor.value_store[value_ref_arg.unique_key]

            value = refcounted_future.future.wait()

            with executor.value_store_lock:
                if refcounted_future.release():
                    executor.value_store.pop(value_ref_arg.unique_key)

            # HACK isend causes the pipeline training to hang
            torch.distributed.send(value, caller_stage, tag=tag)

        # Notify next send that's potentially waiting
        with executor.caller_recv_tag_cv:
            executor.caller_recv_tag[caller_stage] += 1
            executor.caller_recv_tag_cv.notify_all()

    setattr(PipelineDriverBase, "batch_send", safe_batch_send)

    def _hierarchical_init_data_parallel(driver, dp_group_size, dp_pg_cb=None):
        if dp_group_size <= 1:
            logger.info("[root] Data parallel group size <= 1, skipping data parallel initialization")
            return

        n_stages = len(driver.stage_to_executor)
        logger.info(f"[root] Initializing {n_stages} data parallel groups, each of size {dp_group_size}")
        for i in range(0, n_stages, driver.world_size):
            stages = list(range(i, min(i + driver.world_size, n_stages)))
            futs = []
            for stage in stages:
                executor = driver.stage_to_executor[stage]
                futs.append(executor.rpc_async().init_data_parallel(n_stages, dp_group_size, dp_pg_cb))
                _wait_for_all(futs)

    setattr(PipelineDriverBase, "init_data_parallel", _hierarchical_init_data_parallel)


def _install_tensor_meta(fake_node_list, node_list):
    for fake_node, node in zip(fake_node_list, node_list):
        if "tensor_meta" in fake_node.meta:
            node.meta["tensor_meta"] = fake_node.meta["tensor_meta"]


def compile_to_pipe(
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
    traced = to_pippy_graph_module(gm)
    graph = traced.graph

    traced_forward_keys = inspect.signature(traced.forward).parameters

    model_pipe = Pipe._from_traced(
        gm,
        traced,
        multi_use_param_spec,
        output_loss_value_spec=output_loss_value_spec,
    )
    return model_pipe, traced_forward_keys


def _propagate_fake_gm(fake_gm, args, kwargs=dict()):
    """This helper method executes a fake mode shape propogation on a fakified gm
    The fakified gm is stored before TP modifications (which makes shape prop impossible)

    Args:
        args, kwargs: example input to the model

    Returns:
        Tensor meta will be installed on the fake split gm, which can then be retrieved through AutoAccelerateContext
    """
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True) if torch_version() >= (2, 0, 0) else FakeTensorMode()
    complete_args = list(args) + list(kwargs.values())
    fake_args = map_aggregate(
        complete_args,
        lambda a: fake_mode.from_tensor(a.to("meta"))
        if torch_version() >= (2, 0, 0)
        else a.to(next(fake_gm.parameters()).device)
        if isinstance(a, torch.Tensor)
        else None,
    )

    # Each stage_backward node will make a call to autograd.backward
    # We have to default the retain_graph to True to execute the whole graph
    with retain_graph_context():
        sp = RetainGradShapeProp(fake_gm)
        sp.propagate(*fake_args)
    logger.info("finish propagating fake split gm")


def _propagate_fake_split_gm(args, kwargs=dict(), compiler_configs=dict()):
    counter = AutoAccelerateContext.counter
    fake_gm = AutoAccelerateContext.fake_gm[counter]
    compile_with_dynamo = compiler_configs.get("compile_with_dynamo", False)
    multi_use_param_spec = compiler_configs.get("multi_use_param_spec", None)
    output_loss_value_spec = compiler_configs.get("output_loss_value_spec", True)
    _, pipe_ranks = parallel_group_and_ranks("pipe")
    model_pipe, _ = compile_to_pipe(
        fake_gm, pipe_ranks, compile_with_dynamo, multi_use_param_spec, output_loss_value_spec
    )
    _propagate_fake_gm(model_pipe.split_gm, args, kwargs)
    return model_pipe.split_gm


def propagate_fake_split_gm(node_list, args, kwargs=dict(), compiler_configs=dict()):
    """This helper method wrap around the _propagate_fake_split_gm method to install tensor meta on
    the target graph module's node list

    fake_prop_done will be available only if we are using Driver's c10d mode:
        in Driver's c10d mode, redo shape prop if dynamic_shape
    In c10d stage mode:
        this will be called only once, do fake_shape_prop no matter what
    """
    logger.info("Propagating shape across fake split GraphModule")
    dynamic_shape = compiler_configs.get("dynamic_shape", False)
    counter = AutoAccelerateContext.counter
    if not hasattr(AutoAccelerateContext, "fake_prop_done"):
        AutoAccelerateContext.add_ac_attr("fake_prop_done", {counter: False})
    else:
        AutoAccelerateContext.fake_prop_done[counter] = False

    if hasattr(AutoAccelerateContext, "fake_prop_done"):
        do_shape_prop = not AutoAccelerateContext.fake_prop_done[counter] or dynamic_shape
    else:
        do_shape_prop = True
    if do_shape_prop:
        fake_split_gm = _propagate_fake_split_gm(args, kwargs, compiler_configs)
        if not hasattr(AutoAccelerateContext, "fake_split_gm"):
            AutoAccelerateContext.add_ac_attr("fake_split_gm", {counter: fake_split_gm})
        AutoAccelerateContext.fake_prop_done[counter] = True
    else:
        fake_split_gm = AutoAccelerateContext.fake_split_gm[counter]

    _install_tensor_meta(list(fake_split_gm.graph.nodes), node_list)


def construct_output_chunk_spec(args, kwargs=dict()):
    counter = AutoAccelerateContext.counter
    fake_gm = AutoAccelerateContext.fake_gm[counter]
    eval_gm = to_pippy_graph_module(
        copy.deepcopy(fake_gm).eval().to("meta") if torch_version() >= (2, 0, 0) else copy.deepcopy(fake_gm).eval()
    )
    _propagate_fake_gm(eval_gm, args, kwargs)

    output_nodes = [n for n in eval_gm.graph.nodes if n.op == "output"]
    output_metas = tuple([n.meta["tensor_meta"] if "tensor_meta" in n.meta else None for n in output_nodes])

    def extract_chunk_spec(input_):
        if (isinstance(input_, TensorMetadata) or isinstance(input_, FXTensorMetadata)) and len(input_.shape) > 1:
            return TensorChunkSpec(0)
        else:
            return None

    output_chunk_spec = pippy_map_aggregate(
        output_metas,
        extract_chunk_spec,
        lambda a: not (isinstance(a, TensorMetadata) or isinstance(a, FXTensorMetadata)),
    )
    if len(output_chunk_spec) == 1:
        output_chunk_spec = output_chunk_spec[0]
    return output_chunk_spec


def hack_interpreter(compiler_configs=dict()):
    setattr(
        RemoteInterpreter,
        "propagate_shape",
        lambda node_list, args, kwargs: propagate_fake_split_gm(node_list, args, kwargs, compiler_configs),
    )

    def _hack_run_one(interp, node):
        logger.debug(f"[{interp.cur_microbatch}] Issue command to run {node.format_node()}")
        interp.env[node] = Interpreter.run_node(interp, node)

        def wait_for_confirmation(n):
            if isinstance(n, torch._C._distributed_rpc.PyRRef):
                while not n.confirmed_by_owner():
                    pass

        pippy.fx.node.map_aggregate(interp.env[node], wait_for_confirmation)

        if DEBUG and isinstance(interp.env[node], torch._C._distributed_rpc.PyRRef):
            print(node, interp.env[node])
            interp.env[node].to_here()

        # HACK enable c10d for call_function
        if "tensor_meta" in node.meta and isinstance(
            node.meta["tensor_meta"],
            TensorMetadata,
        ):
            val_ref = interp.env[node]
            if isinstance(val_ref, ValueReference):
                val_ref.meta.setdefault("tensor_meta", node.meta["tensor_meta"])

        interp.pc += 1
        return node

    # HACK use c10d for call_function, i.e for backward path
    setattr(RemoteInterpreter, "run_one", _hack_run_one)


def check_split_points(gm):
    graph = gm.graph
    num_split_points = 0
    for node in graph.nodes:
        if (node.op, node.target) == ("call_function", pipe_split):
            num_split_points += 1
    return num_split_points


def to_pippy_graph_module(gm):
    graph = gm.graph
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

    return pippy.fx.GraphModule(gm, new_graph)


def prepare_args_kwargs(example_inputs, forward_keys):
    complete_args = []
    complete_kwargs = {}

    for name, param in forward_keys.items():
        if name in example_inputs:
            if param.default is inspect.Parameter.empty:
                # This is a positional argument
                complete_args.append(example_inputs[name])
            else:
                # This is a keyword argument
                complete_kwargs[name] = example_inputs[name]
        else:
            if param.default is not inspect.Parameter.empty:
                # This is a keyword argument with a default value
                complete_kwargs[name] = param.default

    return complete_args, complete_kwargs
