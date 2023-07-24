import collections
import gc
import os
import pickle
import time
import traceback
from copy import deepcopy

import torch

from atorch.auto.analyser.analyser import get_analyser
from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.device_context import get_device_context
from atorch.auto.dry_runner.dry_runner import get_dryrunner
from atorch.auto.engine.acceleration_engine import AccelerationEngine
from atorch.auto.engine_client import EngineClient
from atorch.auto.model_context import ModelContext
from atorch.auto.opt_lib.optimization_library import SEMIAUTO_STRATEGIES, OptimizationLibrary
from atorch.auto.strategy import Strategy
from atorch.common.constants import AutoAccelerateExtraArgs
from atorch.common.log_utils import default_logger as logger
from atorch.common.util_func import find_free_port
from atorch.data import data_to_device
from atorch.distributed.distributed import (
    create_parallel_group,
    destroy_parallel_group,
    is_distributed,
    local_rank,
    parallel_config,
    rank,
)
from atorch.utils.meta_model_utils import deepcopy_checkpoint_name, reload_meta_module


def model_transform(
    model_context,
    strategy,
    opt_lib,
    apply_wrapper=True,
    create_optim=True,
    create_dataloader=True,
    use_sample_batch=False,
):
    assert not strategy.is_tunable()
    record_user_defined_half_precision_dtype(strategy)
    cpu_offload = False
    for opt in strategy:
        opt_name = opt[0]
        opt_config = opt[1]
        model_context = opt_lib[opt_name].transform(model_context, opt_config)
        if opt_name == "fsdp" and opt_config is not None and opt_config.get("cpu_offload", False) is True:
            cpu_offload = True
    model_context.adjust_wrappers()
    if apply_wrapper:
        model_context.apply_wrappers(is_pre_wrapper=True)
    if create_dataloader:
        model_context.update_dataloader()
    if create_optim:
        model_context.update_optim()
    if use_sample_batch:
        model_context.update_sample_batch()
    if apply_wrapper and create_optim:
        model_context.apply_wrappers(is_pre_wrapper=False)
    if torch.cuda.is_available() and not model_context.gpu_used and not cpu_offload:
        reload_meta_module(model_context.model, torch.device(type="cuda", index=local_rank()), False)
    return model_context


def run_finish_task(model_context, strategy, opt_lib):
    status = True
    result = None
    try:
        result = model_transform(
            model_context,
            strategy,
            opt_lib,
            create_optim=model_context.optim_func is not None,
            create_dataloader=model_context.dataset is not None,
        )
    except Exception as e:
        traceback.print_exc()
        logger.error(f"model_transform failed: {e}")
        status = False
    return status, result


def run_setup_parallel_group_task(parallel_group_info):
    status = True
    # check if parallel_group_info is already setup and the setup can be skipped.
    if parallel_config() == parallel_group_info:
        return status
    try:
        # Upon reconstruction of collective process group,
        # we assume the rpc network on which the Pipeline Parallelism depends does not change.
        # So here we avoid destroying rpc network since it cannot be rebuilt.
        destroy_parallel_group(destroy_rpc=False)
        if parallel_group_info is not None:
            create_parallel_group(parallel_group_info)
    except Exception as e:
        logger.error(f"Setup parallel group failed: {e}")
        status = False
    if is_distributed():
        # Sync, so all processes would finish this task at nearly the same time.
        torch.distributed.barrier()
    return status


def run_analyse_task(model_context, analysis_methods, analyser):
    status = True
    result = None
    try:
        result = analyser.analyse(model_context, analysis_methods)
    except Exception as e:
        logger.error(f"analyse failed: {e}")
        status = False
    return status, result


def run_dryrun_task(model_context, strategy, opt_lib, dry_runner):
    if (
        hasattr(AutoAccelerateContext, "skip_dryrun")
        and AutoAccelerateContext.skip_dryrun[AutoAccelerateContext.counter]
    ):
        return True, dict()
    status = True
    result = None
    mc = None
    try:
        # make a model context copy to keep original model_context
        mc = deepcopy(model_context)
        deepcopy_checkpoint_name(mc.model, model_context.model)
        # TODO: support batchsize search
        batchsize_search = False
        create_dataloader = not batchsize_search and model_context.dataset is not None
        mc = model_transform(
            mc,
            strategy,
            opt_lib,
            create_dataloader=create_dataloader,
            create_optim=model_context.optim_func is not None,
            use_sample_batch=model_context.sample_batch is not None,
        )
        if batchsize_search:
            status, result = dry_runner.dynamic_tune("batchsize", mc, profiling=True)
        else:
            warmup_step_num = int(os.getenv("ATORCH_DRYRUN_WARMUP_STEP", 10))
            profile_step_num = int(os.getenv("ATORCH_DRYRUN_PROFILE_STEP", 15))
            status, result = dry_runner.profile(mc, warmup_step_num, profile_step_num)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"dryrun failed: {e}")
        status = False
    strategy_reset(strategy, opt_lib)
    AutoAccelerateContext.reset()
    if mc is not None:
        # clear gpu if any
        del mc
        torch.cuda.empty_cache()
    return status, result


def run_tune_task(model_context, strategy, opt_lib):
    status = True
    mc = None
    try:
        valid = opt_lib.validate_strategy(strategy)
        if not valid:
            logger.error("Not a valid strategy.")
            status = False
        else:
            mc = deepcopy(model_context)
            deepcopy_checkpoint_name(mc.model, model_context.model)
            record_user_defined_half_precision_dtype(strategy)
            for idx, (name, config, tunable) in enumerate(strategy):
                opt = opt_lib[name]
                if tunable:
                    status, new_config, mc = opt.tune(mc, config, strategy, apply_transform=True)
                    if status is False:
                        break
                    strategy[idx] = (name, new_config, False)
                else:
                    mc = opt.transform(mc, config)
    except Exception as e:
        traceback.print_exc()
        logger.error(f"tune failed: {e}")
        status = False
    strategy_reset(strategy, opt_lib)
    AutoAccelerateContext.reset()
    if mc is not None:
        del mc
        torch.cuda.empty_cache()
    return status, strategy if status else None


def run_task(model_context, task, opt_lib=None, dry_runner=None, analyser=None, verbose=False):
    status = True
    result = None
    if verbose:
        logger.info(f"Process {rank()} run task: {task.type}")
    if task.type == "WAIT":
        time.sleep(2)
    elif task.type == "FINISH":
        status, result = run_finish_task(model_context, task.strategy, opt_lib)
    elif task.type == "SETUP_PARALLEL_GROUP":
        status = run_setup_parallel_group_task(task.parallel_group_info)
        if verbose:
            logger.info(f"SETUP_PARALLEL_GROUP task done, status {status}")
    elif task.type == "ANALYSE":
        status, result = run_analyse_task(model_context, task.analysis_method, analyser)
        if verbose:
            logger.info(f"ANALYSE task ({task.analysis_method}) done, status {status}")
    elif task.type == "DRYRUN":
        status, result = run_dryrun_task(model_context, task.strategy, opt_lib, dry_runner)
        if verbose:
            logger.info(f"DRYRUN task done with {task.strategy}\nResult on rank {rank()}: {result}")
    elif task.type == "TUNE":
        status, result = run_tune_task(model_context, task.strategy, opt_lib)
        if verbose:
            logger.info(f"TUNE task done with {task.strategy}, status {status}")
    elif task.type == "FAIL":
        status = False
    else:
        logger.error(f"Task with type {task.type} is not supported")
        status = False
    gc.collect()
    return status, result


AutoAccelerateResult = collections.namedtuple(
    "AutoAccelerateResult",
    "model, optim, dataloader, loss_func, prepare_input, lr_scheduler",
)


def assemble_result(model_context):
    return AutoAccelerateResult(
        model=model_context.model,
        optim=model_context.optim,
        dataloader=model_context.dataloader,
        loss_func=model_context.loss_func,
        prepare_input=model_context.prepare_input,
        lr_scheduler=model_context.lr_scheduler,
    )


def get_strategy(load_strategy):
    """
    Input args:
        load_strategy: if str, a filename to load pickled strategy from;
                       if bytes, a memory to load pickled strategy from;
                       if isinstance(Strategy), assign directly;
                       if list(str), a list of optimization method names or (name, config).
    Output: status, strategy
      status: Bool, True if loaded strategy successfully
      strategy: the resulting strategy after loading.
    """
    status = True
    data = None
    if isinstance(load_strategy, str):
        try:
            with open(load_strategy, "rb") as fp:
                data = fp.read()
        except Exception as e:
            logger.error(e)
            status = False
    elif isinstance(load_strategy, bytes):
        data = load_strategy
    elif isinstance(load_strategy, Strategy):
        strategy = load_strategy
    elif isinstance(load_strategy, list):
        # a list of opt method names
        strategy = Strategy()
        for item in load_strategy:
            if isinstance(item, str):
                strategy.add_opt((item, None, None))
            elif (isinstance(item, tuple) or isinstance(item, list)) and len(item) == 2:
                # Force tensor parallel to be tunable in any occasions
                strategy.add_opt((item[0], item[1], item[0] in SEMIAUTO_STRATEGIES))
            else:
                logger.error("When use list for load_strategy, should be a list of name or (name, config).")
                status = False
                break
    else:
        logger.error("load_strategy should be str for file, or bytes for memory, or list of name or (name, config)")
        status = False
    if status and data is not None:
        try:
            strategy = pickle.loads(data)
            status = isinstance(strategy, Strategy)
        except Exception:
            logger.error("load_strategy is not in correct strategy format")
            status = False
    if status is False:
        return False, None
    return status, strategy


def save_strategy(strategy, filename):
    """Pickle strategy and save to file with filename"""
    data = pickle.dumps(strategy)
    with open(filename, "wb") as fp:
        fp.write(data)


def adjust_strategy(strategy, device_context, finetune_strategy, opt_lib):
    """Adjust process_mode in strategy according to device_context
    If finetune_strategy, set config to None and tunable to True for tunable methods.
    Also set proper tunable if tunable==None.
    Return: status, adjusted_strategy
    """
    found, parallel_mode = strategy.get_parallel_mode()
    cur_total_process = device_context.node_num * device_context.nproc_per_node
    # adjust data parallel size in parallel_mode
    if parallel_mode is not None:
        st_total_process = 1
        data_parallel_size = 1
        data_parallel_exist = False
        for name, size in parallel_mode[0]:
            st_total_process *= size
            if name == "data":
                data_parallel_size = size
                data_parallel_exist = True
        if st_total_process != cur_total_process:
            if parallel_mode[1] is not None or not data_parallel_exist:
                # if total process num is not equal, we can adjust only for data parallel
                # without custom ranks in process group.
                logger.error("Loaded strategy is not consistent with current device context!")
                return False, None
            if cur_total_process * data_parallel_size % st_total_process != 0:
                logger.error(
                    f"Cannot adjust strategy according to device context: \
                    strategy_total_process={st_total_process} with data_parallel={data_parallel_size}, \
                    device_context_total_process={cur_total_process}"
                )
                return False, None
        new_data_parallel_size = cur_total_process * data_parallel_size // st_total_process
        strategy.adjust_data_parallel(new_data_parallel_size)
    elif found:
        # set parallel mode's config to data parallel only.
        strategy.adjust_data_parallel(cur_total_process)
    if not is_distributed():
        removed_names = strategy.remove_distributed_method(opt_lib)
        if removed_names is not None:
            logger.info("These distributed optimization methods are ignored in non-distributed case: %s", removed_names)

    # reset config for tunable method if finetune_strategy
    if finetune_strategy:
        strategy.reset_config()
    # set tunable value if it is None.
    strategy.set_tunable_value(opt_lib)
    return True, strategy


def record_user_defined_half_precision_dtype(strategy):
    str_dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    for opt in strategy.opt_list:
        opt_name, config = opt[0], opt[1]
        if opt_name in ("amp_native", "half"):
            if opt_name == "amp_native":
                if config is not None:
                    dtype = config.get("dtype")
                    if dtype is None:
                        dtype = torch.float16
                    elif dtype in str_dtype_map:
                        dtype = str_dtype_map[dtype]
                    elif dtype not in str_dtype_map and dtype not in [torch.float16, torch.bfloat16]:
                        raise ValueError(
                            "'amp_native' optimization only support 'fp16', 'bf16', "
                            f"torch.float16 and torch.bfloat16, but got {dtype}"
                        )
                else:
                    dtype = torch.float16
            elif opt_name == "half":
                if config is not None:
                    dtype = str_dtype_map.get(config)
                    if dtype is None:
                        raise ValueError(f"'half' optimization only support 'fp16' and 'bf16', but got {dtype}")
                else:
                    dtype = torch.float16
            if hasattr(AutoAccelerateContext, "half_precision_dtype"):
                if AutoAccelerateContext.counter not in AutoAccelerateContext.half_precision_dtype:
                    AutoAccelerateContext.half_precision_dtype[AutoAccelerateContext.counter] = dtype
                else:
                    if dtype != AutoAccelerateContext.half_precision_dtype[AutoAccelerateContext.counter]:
                        raise ValueError("'amp_native' and 'half' cannot be configured at the same time.")
            else:
                AutoAccelerateContext.add_ac_attr("half_precision_dtype", {AutoAccelerateContext.counter: dtype})


def auto_accelerate(
    model,
    optim_func=None,
    dataset=None,
    loss_func=None,
    prepare_input=None,
    model_input_format=None,
    optim_args=None,
    optim_param_func=None,
    dataloader_args=None,
    distributed_sampler_cls=None,
    excluded=None,
    included=None,
    load_strategy=None,
    lr_scheduler_cls=None,
    lr_scheduler_args=None,
    finetune_strategy=False,
    save_strategy_to_file=None,
    **kargs,
):
    """
    Auto-accelerate the model training. Find a best acceleration strategy and apply it to the model.
    Arguments:
    - model: nn_module defined as a non-distributed model
    - optim_func: optim_func can be a pytorch built-in optimizer function or a user-defined function, with params and
                  optim_args as arguments. such as:
                  def optim_func(parameters, **optim_args):
                      return optim.SGD(parameters, **optim_args)
                  The optimizer will be created by optim_func(model.parameters(), **optim_args).
    - optim_args: if not None, a dict of arguments used for optim, such as:
        optim_args = {"lr": 0.01, "momentum": 0.9}
    - optim_param_func: function returns an optimizer's parameters if users want to specify per-parameter options.
                        such as:
                        def optim_param_func(model):
                            parameters = [{'params': model.base.parameters()},
                                          {'params': model.classifier.parameters(), 'lr': 1e-3}]
                            return parameters
                        If not None, optimizer will be created by
                        optim_func(optim_param_func(model, **kwargs), **optim_args).
    - dataset: dataset for training data IO and preprocessing
    - dataloader_args: args used for creating dataloader, such as {"batch_size": 16, "pin_memory": True}. If user
                       launched a distributed job, batch_size should be total batch size, not per worker batch size.
    - distributed_sampler_cls: if not None, custom distributed sampler with same interface as pytorch's
            DistributedSampler.
    - loss_func: loss function for loss calculation from model input and output, such as:
        def loss_func(input, output):
            loss = nn.MSELoss()
            return loss(input["label"], output)
        This function either returns a loss value, or a list/tuple with the first value as loss.
    - prepare_input: if None, move data generated from dataloader to current process device as model input.
                     if not None, this is a function taken data and device as arguments.
                     call this function on  data generated from dataloader before model input.
                     such as:
                     def prepare_input(data, device):
                         return transform(data["sample"]).to(device), data["label"].to(device)
    - model_input_format: The format in which the input data is passed to the model.
                          If None, data is passed to model by `model(data)`.
                          If unpack_sequence, `model(*data)`.
                          If unpack_dict, `model(**data)`.
    - excluded: if not None, a list of optimization method names, which should NOT be used.
    - included: if not None, a list of optimization method names, which must be used.
    - load_strategy: if not None, load the acceleration stragegy and use it directly.
                     filename to read if str, or
                     a memory to load from if bytes, or
                     a Strategy instance, or
                     list of optimization method name or (name, config).
    - finetune_strategy: if True and load_strategy is not None, finetune the loaded strategy.
    - save_strategy_to_file: if not None, a file name for saving the acceleration strategy.

    Returns: status, result, best_strategy
    - status: a bool indicating if auto_accelerate is successful
    - result: a namedtuple(AutoAccelerateResult) if status is True,
              including model, optim, dataloader, loss_func and prepare_input.
              None if status is False.
    - best_strategy: the best strategy if status is True, otherwise None.
    """
    extra_args = create_extra_args_for_auto_accelerate(**kargs)
    AutoAccelerateContext.counter += 1
    model_context = ModelContext(
        model=model,
        optim_func=optim_func,
        dataset=dataset,
        loss_func=loss_func,
        prepare_input=prepare_input if prepare_input is not None else data_to_device,
        model_input_format=model_input_format,
        optim_args=optim_args,
        optim_param_func=optim_param_func,
        dataloader_args=dataloader_args,
        distributed_sampler_cls=distributed_sampler_cls,
        lr_scheduler_cls=lr_scheduler_cls,
        lr_scheduler_args=lr_scheduler_args,
        extra_args=extra_args,
    )
    device_context = get_device_context()

    verbose = False
    time_limit = None
    if "verbose" in kargs:
        verbose = kargs["verbose"]
    if "time_limit" in kargs:
        time_limit = kargs["time_limit"]

    ignore_dryrun_on_load_strategy = kargs.get("ignore_dryrun_on_load_strategy", False) and load_strategy is not None
    skip_dryrun = if_skip_dryrun(model_context, ignore_dryrun_on_load_strategy)

    if hasattr(AutoAccelerateContext, "skip_dryrun"):
        AutoAccelerateContext.skip_dryrun.update({AutoAccelerateContext.counter: skip_dryrun})
    else:
        AutoAccelerateContext.add_ac_attr("skip_dryrun", {AutoAccelerateContext.counter: skip_dryrun})

    dry_runner = get_dryrunner()
    analyser = get_analyser()
    opt_lib = OptimizationLibrary()

    strategy = None
    if load_strategy is not None:
        status, strategy = get_strategy(load_strategy)
        if not status:
            logger.error("Error in loading strategy.")
            return False, None, None
        if not opt_lib.validate_strategy(strategy):
            logger.error("Loaded strategy is invalid!")
            return False, None, None
        else:
            logger.info(f"Strategy loaded.\n{strategy}")
        # adjust strategy according to device_context and finetune_strategy
        status, strategy = adjust_strategy(strategy, device_context, finetune_strategy, opt_lib)
        if not status:
            return False, None, None
        if not strategy.is_tunable() or not is_distributed():
            # setup_parallel_group if needed
            _, parallel_mode = strategy.get_parallel_mode()
            if parallel_mode is not None:
                status = run_setup_parallel_group_task(parallel_mode)
                if not status:
                    return False, None, None
            if strategy.is_tunable():
                status, strategy = run_tune_task(model_context, strategy, opt_lib)
                if not status:
                    logger.error("Failed to tune the loaded strategy.")
                    return False, None, None
            if skip_dryrun is True:
                logger.info("Dryrun skipped for `ignore_dryrun_on_load_strategy` is True.")
            else:
                # run dryrun task to verify the strategy
                status, result = run_dryrun_task(model_context, strategy, opt_lib, dry_runner)
                if not status:
                    logger.error("Dryrun failed with loaded strategy.")
                    return False, None, None
                else:
                    logger.info(f"Dryrun result on rank {rank()}: {result}")

            # finish task to generate result
            if save_strategy_to_file:
                save_strategy(strategy, save_strategy_to_file)
            status, result = run_finish_task(model_context, strategy, opt_lib)
            assert status
            if strategy is not None:
                AutoAccelerateContext.add_ac_attr(
                    "strategy_opt_names", {AutoAccelerateContext.counter: strategy.opt_names()}
                )
            logger.info(f"Load strategy successfully and ready for use.\n{strategy}")
            setattr(result.model, "_auto_acc_ctx_counter", AutoAccelerateContext.counter)
            return True, assemble_result(result), strategy
    else:
        if (dataset is None and extra_args.get("sample_batch") is None) or optim_func is None:
            logger.error("`load_strategy` should not be None.")
            return False, None, None
    engine = None
    engine_port = 0
    if rank() == 0 or (device_context.node_num == 1 and device_context.nproc_per_node == 1):
        engine_port = find_free_port()
        if strategy is not None:
            strategy = strategy.convert_strategy_to_easydl_format()
        # create AccelerationEngine instance
        engine = AccelerationEngine(
            device_context.context,
            included_opts=included,
            excluded_opts=excluded,
            time_limit=time_limit,
            load_strategy=strategy,
            verbose=verbose,
        )
        engine.start_service(str(engine_port))

    if is_distributed():
        bcast_port = torch.tensor(engine_port, dtype=torch.int32)
        if torch.cuda.is_available():
            device = f"cuda:{local_rank()}"
            bcast_port = bcast_port.to(device=device)
        torch.distributed.broadcast(bcast_port, src=0)
        engine_port = bcast_port.item()
    engine_addr = os.getenv("MASTER_ADDR", "localhost")
    engine_client = EngineClient(engine_addr, str(engine_port))

    while True:
        task = engine_client.get_task()
        if task.type == "FINISH" and save_strategy_to_file:
            save_strategy(task.strategy, save_strategy_to_file)
        status, result = run_task(
            model_context, task, opt_lib=opt_lib, dry_runner=dry_runner, analyser=analyser, verbose=verbose
        )
        if task.type == "FINISH" or task.type == "FAIL":
            if engine is not None:
                engine.tear_down()
                del engine
            if task.type == "FINISH":
                assert status is True
                if task.strategy is not None:
                    AutoAccelerateContext.add_ac_attr(
                        "strategy_opt_names", {AutoAccelerateContext.counter: task.strategy.opt_names()}
                    )
                setattr(result.model, "_auto_acc_ctx_counter", AutoAccelerateContext.counter)
                logger.info(f"auto_accelerate finished!\n{task.strategy}")
                return True, assemble_result(result), task.strategy
            else:
                logger.error("auto_accelerate cannot find a valid strategy to train model!")
                return False, None, None
        elif task.type != "WAIT":
            engine_client.report_task_result(task, status, result)


def strategy_reset(strategy, opt_lib):
    for name, config, _ in strategy:
        opt_lib[name].reset(config)


def create_extra_args_for_auto_accelerate(**kwargs):
    registered_extra_args = AutoAccelerateExtraArgs.all()
    extra_args = {k: v for k, v in kwargs.items() if k in registered_extra_args}
    return extra_args


def if_skip_dryrun(model_context, ignore_dryrun_on_load_strategy=False):
    if ignore_dryrun_on_load_strategy is True:
        logger.info("Found ignore_dryrun_on_load_strategy is True, skip dryrun.")
        return True
    optim_func_str = "optim_func" if model_context.optim_func is None else ""
    loss_func_str = "loss_func" if model_context.loss_func is None else ""
    dataset_str = " dataset" if model_context.dataset is None else ""
    sample_batch_str = "sample_batch" if model_context.sample_batch is None else ""
    none_variable = optim_func_str or loss_func_str or (dataset_str and sample_batch_str)
    if none_variable != "":
        logger.warning(f"Found {none_variable} is None, skip dryrun.")
        return True
    return False
