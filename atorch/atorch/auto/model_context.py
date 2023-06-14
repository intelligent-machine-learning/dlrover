import collections
import copy
import inspect
import os
import types

import torch
from distutils.util import strtobool
from torch.utils.data.distributed import DistributedSampler

import atorch
from atorch.auto.device_context import get_device_context
from atorch.common.log_utils import default_logger as logger
from atorch.data import ShmDataloader, expand_batch_dim, get_sample_batch
from atorch.distributed.distributed import (
    local_rank,
    parallel_group_and_ranks,
    parallel_group_size,
    parallel_rank,
    rank,
)

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


def get_data_partition_rank_and_size():
    data_size = parallel_group_size("data")
    drank = parallel_rank("data")
    if data_size is None:
        data_size = 1
        drank = 0
    zero_size = parallel_group_size("zero")
    if zero_size is not None:
        zrank = parallel_rank("zero")
        drank = drank * zero_size + zrank
        data_size *= zero_size

    return drank, data_size


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

    def update_tp_status(self, status):
        self.tp_status = status

    def convert_to_loss_wrapper(self, amp_config=None):
        pass

    def _check_loss_wrapper(self):
        pass

    def _dynamo_capture_graph(self, input_batch):
        pass

    def capture_compute_graph(self, backend="meta_fx", leaf_modules=None, parallel_config=None):
        pass

    def export_graph_module(self, backend="meta_fx", leaf_modules=None, parallel_config=None):
        pass

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
            sampler = self.distributed_sampler_cls(self.dataset, shuffle=shuffle, num_replicas=ddp_size, rank=rank)
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
        pass

    def create_optim(self):
        """
        Create optimizer from optim_func
        """
        if self.optim_func is None:
            return None
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
        pass

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
            if p.device.type == "cuda":
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
