import collections
import copy
import inspect
import os
import re
import types
from functools import wraps

import torch
from distutils.util import strtobool

try:
    from torch.distributed.fsdp import MixedPrecision
except ImportError:
    MixedPrecision = None
from torch.utils.data.distributed import DistributedSampler

import atorch
from atorch.auto.device_context import get_device_context
from atorch.auto.opt_lib.utils import _propose_leaf_modules, _propose_wrap_cls, to_module_class_by_name
from atorch.common.log_utils import default_logger as logger
from atorch.data import ShmDataloader, expand_batch_dim, get_sample_batch
from atorch.distributed.distributed import (
    get_data_partition_rank_and_size,
    local_rank,
    parallel_group_and_ranks,
    parallel_group_size,
    rank,
)
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device
from atorch.optimizers import BF16Optimizer
from atorch.utils.graph_transform_utils import map_aggregate
from atorch.utils.version import torch_version

try:
    from pippy.IR import LossWrapper
except ImportError:
    LossWrapper = type(object())

from atorch.amp.pipe_amp import _hack_pipe_amp_optimizer, scale_backward_wrapper

try:
    torch.fx.wrap("scale_backward_wrapper")
except ImportError:
    logger.info("FX not supported, some features may be available")


def if_use_shm_dataloader():
    """
    return rank0_global_rank(int), rank(int), group_size(int),
    or None, None, None if not use shm dataloader
    TODO: need to check if fixed batch data size, if not, don't use.
    """
    enabled = strtobool(os.getenv("ENABLE_SHM_DATALOADER", "False"))  # change to default True after TODO
    if not enabled:
        return None, None, None

    world_size = atorch.distributed.world_size()
    ddp_size = parallel_group_size("data")
    if ddp_size is None:
        ddp_size = 1
    if ddp_size == world_size:
        return None, None, None

    # Change this logic if more types of model parallel are added.
    _, model_ranks = parallel_group_and_ranks("model")
    if model_ranks is None:
        return None, None, None

    global_rank = atorch.distributed.rank()
    nproc_per_node = atorch.distributed.nproc_per_node()
    node_index = global_rank // nproc_per_node

    group_size = 0
    rank = None
    rank0_global_rank = None
    for index in range(nproc_per_node * node_index, nproc_per_node * (node_index + 1)):
        if index in model_ranks:
            if rank0_global_rank is None:
                rank0_global_rank = index
            if index == global_rank:
                rank = group_size
            group_size += 1
    if group_size > 1:
        return rank0_global_rank, rank, group_size

    return None, None, None


def get_mc_default_args():
    args = {"use_optim_backward": False, "requires_set_gradient_accumulation_boundary": False}
    return args


class ModelContext(object):
    """
    Model context contains model training related objects,
    such as model, optim, dataset, etc, which are required
    for auto acceleration. Auto-accelerate takes a model context
    as input, and a modified model context as output.
    """

    def __init__(
        self,
        model=None,
        optim_func=None,
        dataset=None,
        loss_func=None,
        prepare_input=None,
        model_input_format=None,
        optim_args=None,
        optim_param_func=None,
        dataloader_args=None,
        distributed_sampler_cls=None,
        lr_scheduler_cls=None,
        lr_scheduler_args=None,
        extra_args=None,
    ):
        self.model = model
        self.optim_func = optim_func
        self.optim = None
        self.dataset = dataset
        self.loss_func = loss_func
        self.prepare_input = prepare_input
        self.model_input_format = self._verify_and_set_input_format(model_input_format)
        self.optim_args = optim_args if optim_args is not None else {}
        self.optim_param_func = optim_param_func
        self.dataloader_args = dataloader_args
        self.dataloader = None
        self.pre_wrappers = {}
        self.post_wrappers = {}
        self.parallel_mode_config = None
        self.pre_wrapper_applied = False
        self.post_wrapper_applied = False
        self.distributed_sampler_cls = (
            DistributedSampler if distributed_sampler_cls is None else distributed_sampler_cls
        )
        self.lr_scheduler_cls = lr_scheduler_cls
        self.lr_scheduler_args = lr_scheduler_args
        self.lr_scheduler = None
        self.extra_args = extra_args if extra_args is not None else {}
        self.sample_batch = self.extra_args.get("sample_batch", None)
        self.sample_batch_size = self.extra_args.get("batch_size", None)
        self.expand_sample_batch = self.extra_args.get("expand_sample_batch", True)
        self._check_data_related_args()
        self.tp_status = False
        self.sampler_seed = self.extra_args.get("sampler_seed", 0)  # pytorch DistributedSamper default seed is 0
        self.args = get_mc_default_args()

    def update_tp_status(self, status):
        self.tp_status = status

    def convert_to_loss_wrapper(self, amp_config=None):
        """This method transforms the model held by this model_context into a loss_wrapper.
        All pipeline involved optimizations need to do such conversion first.
        """
        if amp_config is not None and amp_config["dtype"] != torch.bfloat16:
            _hack_pipe_amp_optimizer()

        if not self._check_loss_wrapper() and self.loss_func is not None:

            class ModelLossWrapper(LossWrapper):
                def __init__(self, module, loss_fn, model_input_format):
                    super().__init__(module, loss_fn)
                    self.model_input_format = model_input_format

                @wraps(type(self.model).forward)
                def forward(self, *args, **kwargs):
                    if self.model_input_format is None:
                        casted_loss = self.loss_fn(*args, self.module(*args, **kwargs))
                    elif self.model_input_format == "unpack_sequence":
                        casted_loss = self.loss_fn(args, self.module(*args, **kwargs))
                    elif self.model_input_format == "unpack_dict":
                        casted_loss = self.loss_fn(kwargs, self.module(*args, **kwargs))
                    else:
                        logger.warning(f"model input format: {self.model_input_format} not recognized")
                        casted_loss = None

                    if amp_config is not None and amp_config["dtype"] != torch.bfloat16:
                        scaled_loss = scale_backward_wrapper(casted_loss)
                    else:
                        scaled_loss = casted_loss
                    return scaled_loss

            loss_wrapper = ModelLossWrapper(
                module=self.model, loss_fn=self.loss_func, model_input_format=self.model_input_format
            )

            self.model = loss_wrapper

        return isinstance(self.model, LossWrapper)

    def _check_loss_wrapper(self):
        return isinstance(self.model, LossWrapper)

    def _dynamo_capture_graph(self, input_batch):
        import torch._dynamo as dynamo

        lowered_device = next(self.model.parameters()).device

        def _lower_tensor_to_device(input_):
            return input_.to(lowered_device) if isinstance(input_, torch.Tensor) else input_

        input_batch = map_aggregate(input_batch, _lower_tensor_to_device)

        sig = inspect.signature(self.model.forward)
        default_names = sig.parameters.keys() - input_batch.keys()
        default_args = {p.name: p.default for p in sig.parameters.values() if p.name in default_names}
        full_args = tuple(input_batch.get(name, default_args.get(name, None)) for name in sig.parameters.keys())
        captured_graph_model = dynamo.export(self.model, *full_args)[0]
        graph = captured_graph_model.graph
        default_placeholders = list()
        orig_input_keys = list(inspect.signature(self.model.forward).parameters.keys())
        # alter the placeholders to align the input_names
        for node in graph.nodes:
            if node.op == "placeholder" and node.name not in orig_input_keys:
                arg_index = int(re.findall(r"\d+", node.name)[0])
                orig_node_name = orig_input_keys[arg_index]
                node.name = orig_node_name
                node.target = orig_node_name
                if node.name not in input_batch.keys():
                    default_placeholders.append(node)
        for node in default_placeholders:
            graph.erase_node(node)

        return captured_graph_model

    def capture_compute_graph(self, backend="meta_fx", leaf_modules=None, parallel_config=None):
        if leaf_modules is not None and backend != "meta_fx":
            logger.warning(f"leaf modules registration is not supported with backend: {backend}")

        input_batch = self.get_one_input_batch(need_model_input_correspondence=True, parallel_config=parallel_config)

        if backend == "dynamo":
            try:
                captured_graph_model = self._dynamo_capture_graph(input_batch)
                self.model = captured_graph_model[0]
                return captured_graph_model[0].graph
            except Exception as e:
                logger.warning(f"tracing failed with exception: {e}")
                logger.warning("torch._dynamo.export does not support dynamic loop")
                backend = "meta_fx"

        if backend == "meta_fx" or backend == "fx":
            meta = backend == "meta_fx"
            if not meta:
                from torch.fx import Tracer

                tracer = Tracer()
                try:
                    graph = tracer.trace(self.model)
                    return graph
                except Exception as e:
                    logger.warning(f"tracing failed with exception: {e}, try meta tracer instead")

            from atorch.utils.tracer import MetaTracer

            tracer = MetaTracer()
            tracer.register_leaf_modules(leaf_modules)
            graph = tracer.trace(self.model, input_batch)
            return graph

    def export_graph_module(self, backend="meta_fx", leaf_modules=None, parallel_config=None):
        # This captures the model's compute graph and export a graph module,
        # yet self.model is not converted.
        graph = self.capture_compute_graph(backend=backend, leaf_modules=leaf_modules, parallel_config=parallel_config)
        return torch.fx.GraphModule(self.model, graph)

    def create_dataloader(self, extra_args=None):
        """
        Create a dataloader and return it
        """
        if self.dataset is None:
            return None
        args = self.dataloader_args
        if args is None:
            args = extra_args if extra_args else {}
        elif extra_args:
            args.update(extra_args)
        # TODO: find a best num_workers value
        recommended_num_workers = min(get_device_context().cpu_num_per_node // get_device_context().nproc_per_node, 32)
        if "num_workers" not in args:
            args["num_workers"] = recommended_num_workers
        else:
            # Do not update `num_workers` set by user. Because `num_workers` may be set to 0 to debug Dataset.
            if args["num_workers"] != recommended_num_workers:
                logger.info(
                    f"Found Dataloader's `num_workers` is {args['num_workers']}. It is highly recommended that "
                    f"set `num_workers` to {recommended_num_workers} in order to accelerate data preprocessing "
                    f"and avoid IO bottleneck."
                )
        rank, ddp_size = get_data_partition_rank_and_size()
        sampler = None
        if ddp_size > 1:
            shuffle = True
            if "shuffle" in args:
                shuffle = bool(args["shuffle"])
                if shuffle:
                    args["shuffle"] = False
            sampler = self.distributed_sampler_cls(
                self.dataset, shuffle=shuffle, num_replicas=ddp_size, rank=rank, seed=self.sampler_seed
            )
            # strong scaling, so adjust batchsize
            if "batch_size" in args:
                ori_batchsize = args["batch_size"]
                args["batch_size"] = ori_batchsize // ddp_size
                if args["batch_size"] == 0:
                    args["batch_size"] = 1
                total_batchsize = args["batch_size"] * ddp_size
                if total_batchsize != ori_batchsize:
                    logger.warning(
                        "Batchsize={} is not a multiple of data parallel size {}, adjusted to {}.".format(
                            ori_batchsize, ddp_size, total_batchsize
                        )
                    )

        # TODO: if sequence parallel is used, supports only one rank reads/processes data.
        rank0_global_rank, rank, group_size = if_use_shm_dataloader()
        if rank0_global_rank is not None:
            dataloader_args = args
            dataloader_args["sampler"] = sampler
            dataloader_args["num_workers"] = min(dataloader_args["num_workers"] * group_size, 36)
            dataloader_args["drop_last"] = True  # not support different batch size for now.
            io_timeout = int(os.getenv("SHM_DATALOADER_IO_TIMEOUT", 30))
            initialize_timeout = int(os.getenv("SHM_DATALOADER_INIT_TIMEOUT", 120))
            shm_name_prefix = f"shm_dataloader_{rank0_global_rank}_"
            dataloader = ShmDataloader(
                self.dataset,
                dataloader_args,
                rank=rank,
                group_size=group_size,
                io_timeout=io_timeout,
                initialize_timeout=initialize_timeout,
                need_sync_write=False,
                shm_data_size=2,
                shm_name_prefix=shm_name_prefix,
            )
        else:
            dataloader = torch.utils.data.DataLoader(self.dataset, sampler=sampler, **args)

        def set_epoch(self, epoch):
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epoch)

        dataloader.set_epoch = types.MethodType(set_epoch, dataloader)
        return dataloader

    def update_dataloader(self, extra_args=None):
        """
        Update dataloader
        """
        self.dataloader = self.create_dataloader(extra_args)

    def check_pipe_model(self):
        # import pippy related modules here to avoid PiPPy initialization messes up env
        from atorch.modules.distributed_modules.compilers import DeviceSafeDriver, SafeStage

        return isinstance(self.model, (DeviceSafeDriver, SafeStage))

    def create_optim(self):
        """
        Create optimizer from optim_func
        """
        if self.optim_func is None:
            return None

        # model must on cuda device before calling OSS (zero2)
        if "zero2" in self.post_wrappers or "ds_zero" in self.post_wrappers:
            is_cuda = next(self.model.parameters()).is_cuda
            if torch.cuda.is_available() and not is_cuda:
                if local_rank() is not None:
                    device = torch.device(type="cuda", index=local_rank())
                else:
                    device = torch.device(type="cuda")
                materialize_modules_to_device(self.model, device)

        if "ds_zero" in self.post_wrappers:
            # sync model parameters/buffers.
            module_states = []
            for param in self.model.parameters():
                module_states.append(param.detach())

            for buffer in self.model.buffers():
                module_states.append(buffer.detach())

            src = 0
            process_group, ranks = parallel_group_and_ranks("zero")
            if process_group is None:
                process_group, ranks = parallel_group_and_ranks("data")
            if ranks is not None:
                src = ranks[0]
            torch.distributed._broadcast_coalesced(process_group, module_states, int(250 * 1024 * 1024), src)

        if not self.check_pipe_model():
            if not self.optim_param_func:
                optim = self.optim_func(self.model.parameters(), **self.optim_args)
            else:
                optim = self.optim_func(self.optim_param_func(self.model), **self.optim_args)
        else:
            optim = self.model.instantiate_optimizer(self.optim_func, **self.optim_args)
            if self.optim_param_func:
                logger.warning(
                    "model parameter retrieval is handled by pipe stage executor, optim_param_func will not take effect"
                )
            logger.info(f"successfully construct optimizer at {rank()}")

        # Use BF16Optimizer if using half with bf16 and not using ds zero
        if (
            "half" in self.pre_wrappers
            and self.pre_wrappers["half"][1] == "bf16"
            and "ds_zero" not in self.post_wrappers
            and "zero2" not in self.post_wrappers
            and "fsdp" not in self.pre_wrappers
            and "ds_3d_parallel" not in self.post_wrappers
        ):
            is_cuda = next(self.model.parameters()).is_cuda
            # model must on cuda device before calling BF16Optimizer
            if torch.cuda.is_available() and not is_cuda:
                if local_rank() is not None:
                    device = torch.device(type="cuda", index=local_rank())
                else:
                    device = torch.device(type="cuda")
                materialize_modules_to_device(self.model, device)
            optim = BF16Optimizer(optim)
        return optim

    def _create_lr_scheduler(self):
        # This method should only be called after the creat_optim call
        if not self.check_pipe_model():
            logger.warning("Not a pipe model, do not handle lrs creation")
            return None
        else:
            return self.model.instantiate_lr_scheduler(self.lr_scheduler_cls, **self.lr_scheduler_args)

    def update_optim(self):
        """
        Update optimizer
        """
        self.optim = self.create_optim()
        if self.check_pipe_model() and self.lr_scheduler_cls is not None:
            self.lr_scheduler = self._create_lr_scheduler()
            logger.info(f"finish instantiate lr_scheduler {self.lr_scheduler} at {rank()}")

    def apply_wrappers(self, is_pre_wrapper=True):
        if (is_pre_wrapper and self.pre_wrapper_applied) or (not is_pre_wrapper and self.post_wrapper_applied):
            return
        wrappers = self.pre_wrappers if is_pre_wrapper else self.post_wrappers
        # Since mp wrapper will add tp and pipe wrappers into wrappers,
        # we have to take it out and execute it separately
        (wrapper_func, wrapper_config) = wrappers.pop("mp", (None, None))
        if wrapper_func is not None:
            wrapper_func(self, "mp", wrapper_config)
            # Since MP wrapper will add some more wrappers, we will call adjust wrappers again.
            self.adjust_wrappers()

        for name in wrappers.keys():
            (wrapper_func, wrapper_config) = wrappers[name]
            wrapper_func(self, name, wrapper_config)
        if is_pre_wrapper:
            self.pre_wrapper_applied = True
        else:
            self.post_wrapper_applied = True

    def adjust_wrappers(self):
        """Adjust wrappers, remove incompatible wrappers.
        pre-wrapper order:
         - half
         - module_replace
         - fp8
         - fsdp
         - native_dynamo (if not moved to post-wrapper)
         post-wrapper list (no order):
         - zero2
         - ds_zero
         - amp_native (ignored if half exists)
         - checkpoint
         - ds_3d_parallel
         - ddp
        """
        if "amp_native" in self.post_wrappers and "half" in self.pre_wrappers:
            logger.info("Both half and amp_native are used, amp_native is removed.")
            self.post_wrappers.pop("amp_native")

        # Pipeline parallelism is conditionally compatible with tensor parallel compilation.
        # To mix pipe and tp, use MixedParallelOptimization instead.
        # We assume MP does not coexists with Pipe/TP
        pipe_wrapper_exist = "pipe" in self.pre_wrappers
        mp_wrapper_exist = "mp" in self.pre_wrappers
        if pipe_wrapper_exist or mp_wrapper_exist:
            logger.info("Found pipe wrapper, remove all ddp related wrappers")
            if "zero2" in self.pre_wrappers:
                self.pre_wrappers.pop("zero2")
            if "fsdp" in self.pre_wrappers:
                self.pre_wrappers.pop("fsdp")

            # DDP is supported and handled internally by PiPPy.
            if "ddp" in self.post_wrappers:
                self.post_wrappers.pop("ddp")

        if pipe_wrapper_exist:
            # Order of execution:
            # Pipe (with amp, to graph only) -> TP -> Checkpoint (if not Pipe)
            # -> module replace -> amp (if no Pipe) -> Pipe (to driver)
            if "checkpoint" in self.post_wrappers:
                ckpt_wrapper = self.post_wrappers.pop("checkpoint")
                pipe_config = self.pre_wrappers["pipe"][1] or {}
                ckpt_wrap_cls = ckpt_wrapper[1]
                pipe_wrap_cls = set(to_module_class_by_name(self.model, ckpt_wrap_cls))
                leaf_modules = _propose_leaf_modules(pipe_wrap_cls)
                pipe_config["compiler_configs"]["checkpoint"] = True
                pipe_config["compiler_configs"]["leaf_modules"] = leaf_modules
                self.pre_wrappers["pipe"] = (
                    self.pre_wrappers["pipe"][0],
                    pipe_config,
                )

            if "amp_native" in self.post_wrappers:
                amp_wrapper = self.post_wrappers.pop("amp_native")
                amp_config = amp_wrapper[1] or None
                pipe_config = self.pre_wrappers["pipe"][1] or {}
                pipe_config["compiler_configs"]["amp_config"] = amp_config
                self.pre_wrappers["pipe"] = (
                    self.pre_wrappers["pipe"][0],
                    pipe_config,
                )

            if "module_replace" in self.pre_wrappers:
                self.pre_wrappers.pop("module_replace")
                pipe_config = self.pre_wrappers["pipe"][1] or {}
                pipe_config["compiler_configs"]["module_replace"] = True
                self.pre_wrappers["pipe"] = (
                    self.pre_wrappers["pipe"][0],
                    pipe_config,
                )

        # FIXME Allow mixing of DDP/ZeRO with MP?
        if mp_wrapper_exist:
            mp_config = self.pre_wrappers["mp"][1] or {}
            pipe_config = mp_config["pipe_config"]
            if pipe_config is not None:
                if "checkpoint" in self.post_wrappers:
                    ckpt_wrapper = self.post_wrappers.pop("checkpoint")
                    ckpt_wrap_cls = ckpt_wrapper[1]
                    pipe_wrap_cls = set(to_module_class_by_name(self.model, ckpt_wrap_cls))
                    leaf_modules = _propose_leaf_modules(pipe_wrap_cls)
                    pipe_config["compiler_configs"]["checkpoint"] = True
                    pipe_config["compiler_configs"]["leaf_modules"] = leaf_modules

                if "amp_native" in self.post_wrappers:
                    amp_wrapper = self.post_wrappers.pop("amp_native")
                    amp_config = amp_wrapper[1] or None
                    pipe_config["compiler_configs"]["amp_config"] = amp_config

                if "module_replace" in self.pre_wrappers:
                    self.pre_wrappers.pop("module_replace")
                    pipe_config["compiler_configs"]["module_replace"] = True

                mp_config["pipe_config"] = pipe_config

                self.pre_wrappers["mp"] = (
                    self.pre_wrappers["mp"][0],
                    mp_config,
                )

        ddp_wrapper_exist = "ddp" in self.post_wrappers
        ds_zero_wrapper_exist = "ds_zero" in self.post_wrappers
        ds_3d_parallel_wrapper_exist = "ds_3d_parallel" in self.post_wrappers
        fairscale_zero2_wrapper_exist = "zero2" in self.post_wrappers
        fsdp_wrapper_exist = "fsdp" in self.pre_wrappers or "zero2" in self.pre_wrappers
        tensor_parallel_wrapper_exist = "tp" in self.pre_wrappers
        ckpt_wrapper_exist = "checkpoint" in self.post_wrappers
        native_dynamo_wrapper_exist = "native_dynamo" in self.pre_wrappers

        # remove ddp wrapper when using zero2
        if ddp_wrapper_exist and (
            fairscale_zero2_wrapper_exist or fsdp_wrapper_exist or ds_zero_wrapper_exist or ds_3d_parallel_wrapper_exist
        ):
            logger.info("Found Zero, ds_3d_parallel, or pipe wrapper, remove ddp wrapper.")
            self.post_wrappers.pop("ddp")
            ddp_wrapper_exist = False
        if fsdp_wrapper_exist and "amp_native" in self.post_wrappers:
            logger.info("Found fsdp and amp_native wrapper, turn on mixed_precision in FSDP")
            _, amp_native_config = self.post_wrappers["amp_native"]
            fp16_dtype = amp_native_config.get("dtype", torch.float16)
            mixed_precision_param = (
                MixedPrecision(param_dtype=fp16_dtype, reduce_dtype=fp16_dtype, buffer_dtype=fp16_dtype)
                if MixedPrecision
                else True
            )
            config = self.pre_wrappers["fsdp"][1] or {}
            config["mixed_precision"] = mixed_precision_param
            self.pre_wrappers["fsdp"] = (
                self.pre_wrappers["fsdp"][0],
                config,
            )

        # move dynamo_native wrapper behind ddp or fsdp (fsdp will adjusted later)
        # Note that dynamo_native wrapper and fsdp wrapper are pre-wrappers while ddp wrapper is a post-wrapper.
        if native_dynamo_wrapper_exist and ddp_wrapper_exist and not fsdp_wrapper_exist:
            # ddp wrapper is a post-wrapper. Popping dynamo_native wrapper from pre-wrappers
            # then insert it after ddp wrapper.
            post_wrappers_list = []
            ddp_wrapper_index = -1
            for i, (wrapper_name, v) in enumerate(self.post_wrappers.items()):
                post_wrappers_list.append((wrapper_name, v))
                if wrapper_name == "ddp":
                    ddp_wrapper_index = i
                native_dynamo_wrapper = self.pre_wrappers.pop("native_dynamo")
                post_wrappers_list.insert(ddp_wrapper_index + 1, ("native_dynamo", native_dynamo_wrapper))
            self.post_wrappers = dict(post_wrappers_list)

        if tensor_parallel_wrapper_exist:
            wrap_cls = None
            if fsdp_wrapper_exist and torch_version() >= (1, 12, 0):
                fsdp_wrapper = self.pre_wrappers["fsdp"]
                fsdp_wrapper = list(fsdp_wrapper)
                if fsdp_wrapper[1] is None:
                    fsdp_wrapper[1] = dict()

                fsdp_config = fsdp_wrapper[1]
                if wrap_cls is None:
                    wrap_cls = set(to_module_class_by_name(self.model, fsdp_config.get("atorch_wrap_cls", set())))
                else:
                    wrap_cls = wrap_cls & set(
                        to_module_class_by_name(self.model, fsdp_config.get("atorch_wrap_cls", set()))
                    )

            if ckpt_wrapper_exist:
                ckpt_wrapper = self.post_wrappers["checkpoint"]
                ckpt_wrapper = list(ckpt_wrapper)
                if ckpt_wrapper[1] is None:
                    ckpt_wrapper[1] = tuple()
                ckpt_wrap_cls = ckpt_wrapper[1]
                if wrap_cls is None:
                    wrap_cls = set(to_module_class_by_name(self.model, ckpt_wrap_cls))
                else:
                    wrap_cls = set(to_module_class_by_name(self.model, ckpt_wrap_cls)) & wrap_cls

            leaf_modules = _propose_leaf_modules(wrap_cls)
            auto_wrap_cls = _propose_wrap_cls(leaf_modules)

            if fsdp_wrapper_exist and torch_version() >= (1, 12, 0):
                if "atorch_wrap_cls" in fsdp_config:
                    if auto_wrap_cls is not None:
                        fsdp_config["atorch_wrap_cls"] = auto_wrap_cls
                    else:
                        fsdp_config.pop("atorch_wrap_cls")

                    fsdp_wrapper[1] = fsdp_config
                    self.pre_wrappers["fsdp"] = tuple(fsdp_wrapper)

            if ckpt_wrapper_exist:
                if auto_wrap_cls is not None:
                    ckpt_wrapper[1] = tuple(auto_wrap_cls)
                    self.post_wrappers["checkpoint"] = tuple(ckpt_wrapper)
                else:
                    # in this case module structure should have been broken, nothing to checkpoint on
                    self.post_wrappers.pop("checkpoint")

            # let tensor parallel wrapper be the first, make sure meta models fully reloaded
            tensor_parallel_wrapper_item = ("tp", self.pre_wrappers["tp"])
            wrappers_list = list(self.pre_wrappers.items())
            tensor_parallel_idx = wrappers_list.index(tensor_parallel_wrapper_item)
            wrappers_list.pop(tensor_parallel_idx)
            # wrapper item and wrapper are all tuples
            tensor_parallel_wrapper_item = list(tensor_parallel_wrapper_item)
            tensor_parallel_wrapper_item[1] = list(tensor_parallel_wrapper_item[1])
            tensor_parallel_wrapper_item[1][1]["leaf_modules"] = leaf_modules
            if fsdp_wrapper_exist or pipe_wrapper_exist:
                tensor_parallel_wrapper_item[1][1]["defer_init"] = True
            tensor_parallel_wrapper_item[1] = tuple(tensor_parallel_wrapper_item[1])
            tensor_parallel_wrapper_item = tuple(tensor_parallel_wrapper_item)
            wrappers_list.insert(0, tensor_parallel_wrapper_item)
            self.pre_wrappers = dict(wrappers_list)

            # TP checkpointing needs amp_config explicitly to take effect, HACK here
            if "amp_native" in self.post_wrappers and "checkpoint" in self.post_wrappers:
                amp_wrapper = self.post_wrappers.pop("amp_native")
                amp_config = amp_wrapper[1] or None
                from atorch.modules.distributed_modules.activation_checkpointing import _insert_amp_config_for_tp_ckpt

                _insert_amp_config_for_tp_ckpt(amp_config)

        # adjust pre_wrapper order
        order_wrapper_name = ["half", "module_replace", "sequence_parallel", "fp8", "fsdp", "native_dynamo"]
        match_names = []
        for name in self.pre_wrappers:
            if name in order_wrapper_name:
                match_names.append(name)
        if len(order_wrapper_name) > 1:
            ordered_wrappers = {}
            for name in self.pre_wrappers:
                if name not in match_names:
                    ordered_wrappers[name] = self.pre_wrappers[name]
                else:
                    for wrapper_name in order_wrapper_name:
                        if wrapper_name in match_names:
                            match_names.remove(wrapper_name)
                            ordered_wrappers[wrapper_name] = self.pre_wrappers[wrapper_name]
            self.pre_wrappers = ordered_wrappers

    def add_wrapper(self, wrapper_name, wrapper_func, wrapper_config=None, is_pre_wrapper=True):
        """
        wrapper_name: name of the wrapper
        wrapper_func: the function to apply this wrapper
        wrapper_config: the config of this wrapper
        is_pre_wrapper: True if pre_wrapper, False if post_wrapper
            pre_wrapper will apply before optim/dataloader instance creation
            post_wrapper will apply after optim/dataloader instance creation
        Save the wrapper in wrappers(dict), use wrapper_name as key
        and (wrapper_func, wrapper_config) as value.
        """
        wrappers = self.pre_wrappers if is_pre_wrapper else self.post_wrappers
        wrappers[wrapper_name] = (wrapper_func, wrapper_config)

    def _verify_and_set_input_format(self, model_input_format):
        assert model_input_format in (
            None,
            "unpack_sequence",
            "unpack_dict",
        ), f"model_input_format should be `None`, 'unpack_sequence', 'unpack_dict' but got {model_input_format}"
        return model_input_format

    @property
    def gpu_used(self):
        """
        Return True if any of model's parameters on gpu
        """
        # PipeModel should be completely handled by itself
        if self.check_pipe_model():
            return True
        for p in self.model.parameters():
            if p.is_cuda:
                return True
        return False

    @property
    def find_unused_parameters(self):
        if "find_unused_parameters" in self.extra_args:
            return self.extra_args["find_unused_parameters"]
        return False

    def get_input_batch_size(self, parallel_config=None):
        tmp_args = copy.deepcopy(self.dataloader_args)
        if tmp_args is None:
            tmp_args = {}
        parallel_config = parallel_config if parallel_config else dict()
        ddp_size = parallel_config.get("ddp_size", get_data_partition_rank_and_size()[1])
        if ddp_size > 1 and "batch_size" in tmp_args:
            tmp_args["batch_size"] = tmp_args["batch_size"] // ddp_size

        chunks = parallel_config.get("chunks", 1)
        if chunks is not None:
            tmp_args["batch_size"] = tmp_args["batch_size"] // chunks
        return tmp_args["batch_size"]

    def get_one_input_batch(self, need_model_input_correspondence=True, parallel_config=None):
        """Get one batch of input data

        Args:
            need_model_input_correspondence (bool): if True, returns a dict with keys corresponding
                to each of the input arguments to model.forward. If False, returns the raw output of
                prepare_input
        """
        tmp_args = copy.deepcopy(self.dataloader_args)
        if tmp_args is None:
            tmp_args = {}
        tmp_args["persistent_workers"] = False
        parallel_config = parallel_config if parallel_config else dict()
        ddp_size = parallel_config.get("ddp_size", get_data_partition_rank_and_size()[1])
        if ddp_size > 1:
            if "batch_size" in tmp_args:
                tmp_args["batch_size"] = tmp_args["batch_size"] // ddp_size
            if self.sample_batch_size is not None:
                self.sample_batch_size = self.sample_batch_size // ddp_size
        chunks = parallel_config.get("chunks", 1)
        if chunks is not None:
            if "batch_size" in tmp_args:
                tmp_args["batch_size"] = tmp_args["batch_size"] // chunks
            if self.sample_batch_size is not None:
                self.sample_batch_size = self.sample_batch_size // chunks
        if self.dataset is not None:
            data = get_sample_batch(self.dataset, tmp_args)
        elif self.sample_batch is not None:
            data = (
                expand_batch_dim(self.sample_batch, self.sample_batch_size)
                if self.expand_sample_batch
                else self.sample_batch
            )
        else:
            logger.warning(
                "Both dataset and sample_batch are None. Return None. If you want to use tensor parallel "
                "or pipeline parallel, set either dataset or sample_batch."
            )
            return None

        if self.prepare_input is not None:
            device = "cuda:{}".format(local_rank()) if torch.cuda.is_available() else "cpu"
            data = self.prepare_input(data, device)
        if not need_model_input_correspondence or self.model_input_format == "unpack_dict":
            return data
        else:
            input_names = list(inspect.signature(self.model.forward).parameters.keys())
            if self.model_input_format == "unpack_sequence":
                return {input_name: data_item for input_name, data_item in zip(input_names, data)}
            else:
                return {input_names[0]: data}

    @staticmethod
    def get_loss_from_loss_func_output(output):
        if isinstance(output, collections.abc.Sequence):
            return output[0]
        return output

    def _check_data_related_args(self):
        if self.sample_batch is not None:
            if self.dataset is not None:
                logger.warning(
                    f"`{self.dataset}` and `sample_batch` are mutually excluded. If you want"
                    f" to create a dataloader, do not pass `sample_batch` args. Otherwise, set "
                    f"`{self.dataset}` to None. `sample_batch` will be ignored."
                )
                self.sample_batch = None
                return
            assert self.sample_batch_size is not None, "Did not get the batch size of `sample_batch`."

    def update_sample_batch(self):
        if self.sample_batch is not None and self.sample_batch_size is not None:
            _, ddp_size = get_data_partition_rank_and_size()
            self.sample_batch_size = self.sample_batch_size // ddp_size
            if self.sample_batch_size == 0:
                self.sample_batch_size = 1
            self.sample_batch = expand_batch_dim(self.sample_batch, batch_size=self.sample_batch_size)
