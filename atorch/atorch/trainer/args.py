import contextlib
import os.path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING
from transformers.trainer_utils import IntervalStrategy
from transformers.utils import cached_property, is_torch_available, requires_backends

from atorch.common.log_utils import default_logger as logger
from atorch.trainer.state import AtorchAcceleratorState
from atorch.utils.dataclass_utils import AutoMapperExtraConfigs, DataclassMixin, DynamicDataClass
from atorch.utils.import_util import is_torch_npu_available

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Attention to developers!!!
# If you want to add a new mapping to the following dict, you should ensure the arg in AtorchTrainingArgs
# and that in Megatron args have same default value.
COMMON_TO_MEGATRON_ARG_MAP = {
    "bf16": "bf16",
    "fp16": "fp16",
    "output_dir": "save",
    "resume_from_checkpoint": "load",
    "save_steps": "save_interval",
    "logging_steps": "log_interval",
    "eval_steps": "eval_interval",
    "tensorboard_dir": "tensorboard_dir",
    # torch init process
    # Comment the following two lines temporarily.
    # "ddp_backend": "distributed_backend",
    # "ddp_timeout": "distributed_timeout_minutes",  # seconds: minutes
}


@dataclass
class AtorchTrainingArgs(DataclassMixin, AutoMapperExtraConfigs):
    profiler_type: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Select a profiler platform. "hw": torch NPU profiler; "hw_dp": torch NPU dynamic profiler; '
            '"nv": origin torch profiler.',
            "choices": [None, "hw", "hw_dp", "nv"],
        },
    )
    profiler_file_path: Optional[str] = field(
        default=None,
    )
    profiler_config: dict = field(default_factory=dict)
    profiler_schedule_skip_first: int = field(
        default=20, metadata={"help": "Torch profiler schedule 'skip_first' arg."}
    )
    profiler_schedule_wait: int = field(default=1, metadata={"help": "Torch profiler schedule 'wait' arg."})
    profiler_schedule_warmup: int = field(default=1, metadata={"help": "Torch profiler schedule 'warmup' arg."})
    profiler_schedule_active: int = field(default=1, metadata={"help": "Torch profiler schedule 'active' arg."})
    profiler_schedule_repeat: int = field(default=1, metadata={"help": "Torch profiler schedule 'repeat' arg."})
    dynamic_profiler_config_path: Optional[str] = field(
        default=None, metadata={"help": "The config directory when using torch NPU dynamic profiler."}
    )

    flash_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Whether use async checkpoint saving. Async saving will"
            "not block calculation process during ckpt saving."
        },
    )

    save_total_limit: int = field(
        default=None,
        metadata={"help": "Max save ckpt versions. 'None' means will keep all the checkpoints."},
    )

    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    rng_types: Optional[List[str]] = field(default=None)

    dispatch_batches: Optional[bool] = field(default=None)

    resume_from_checkpoint: Union[str, None] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})

    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/NPU/PPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/NPU/PPU core/CPU for evaluation."},
    )

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_ratio: float = field(
        default=0.0,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    # Half precision
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (no_cuda). This is an experimental API and it may change."
            )
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"},
    )

    load_model_func_kwargs: dict = field(default_factory=dict)

    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    checkpoint_save_limit: Optional[int] = field(default=3, metadata={"help": ""})

    distributed_type: str = field(
        default="ddp",
        metadata={
            "help": "Select a distributed training method.",
            "choices": [
                "ddp",
                "megatron",
            ],  # TODO: to support "fsdp", "deepspeed" and "auto"
        },
    )

    ddp_backend: Optional[str] = field(
        default="nccl",
        metadata={
            "help": "The backend to be used for distributed training",
            "choices": ["nccl", "gloo", "hccl"],
        },
    )

    ddp_timeout: Optional[int] = field(
        default=1800,
        metadata={
            "help": (
                "Timeout for torch.distributed. Overrides the default timeout for "
                "distributed training (value should be given in seconds)."
            )
        },
    )

    # Args about optimizer.
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    safe_serialization: Optional[bool] = field(default=True)

    mixed_precision: Optional[str] = field(default=None)

    save_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )

    save_steps: int = field(
        default=500,
        metadata={"help": ("Save checkpoint every X steps. Should be an integer.")},
    )

    # Evaluation
    evaluation_strategy: Union[IntervalStrategy, str] = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )

    eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": ("Run an evaluation every X steps.")},
    )

    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": (
                "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the"
                " evaluation_strategy."
            )
        },
    )

    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})

    map_location: Optional[str] = field(default=None)

    # Args about logging
    disable_tqdm: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether or not to disable the tqdm progress bars."},
    )

    tensorboard_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})

    logging_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )

    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})

    logging_steps: int = field(
        default=500,
        metadata={"help": ("Log every X updates steps.")},
    )

    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})

    log_params_std: bool = field(default=False, metadata={"help": "If set, calculate and log parameters std."})

    log_grad_diff_for_debug: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to log the difference in gradient norm before and after scaling with the spike loss ratio."
            )
        },
    )

    # Args about data
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )

    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )

    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."},
    )

    def validate_megatron_args(self):
        if self.distributed_type not in ["megatron", "mindspeed"]:
            return

        if self.resume_from_checkpoint is not None:
            assert not os.path.basename(self.resume_from_checkpoint).startswith("iter"), (
                "megatron and mindspeed support only resume from latest checkpoint, make sure you set the "
                "resume_from_checkpoint to the out_dir of the checkpoints instead of the iter_XXX subfolder."
            )

    def __post_init__(self):
        self._megatron_args = None

        if self.save_total_limit is not None:
            assert self.save_total_limit > 0, "save_total_limit should greater than 0 or None"

        if self.resume_from_checkpoint and self.overwrite_output_dir:
            raise ValueError(
                "Please don't set resume_from_checkpoint and overwrite_output_dir at the same time, this "
                "may leads to unexpected behaviour. overwrite_output_dir implies training from scratch, "
                "while resume_from_checkpoint implies training basing on a existing checkpoint."
            )

        if self.resume_from_checkpoint is not None:
            try:
                resume_from_checkpoint_is_not_empty = len(os.listdir(self.resume_from_checkpoint)) > 0
                assert (
                    resume_from_checkpoint_is_not_empty
                ), f"unable to resume from an empty checkpoint {self.resume_from_checkpoint}, please check your config"
            except FileNotFoundError:
                raise ValueError(
                    f"does not find {self.resume_from_checkpoint} as a ckpt dir to resume, please check your config."
                )

        # Not resuming from checkpoint, user have to make sure that the output dir must be empty,
        # otherwise ckpts might be overwritten.
        # TODO: move the following code out of __post_init__()
        if self.resume_from_checkpoint is None and not self.overwrite_output_dir:
            output_dir_is_empty = len(os.listdir(self.output_dir)) == 0
            assert output_dir_is_empty, (
                "output dir is not empty, you have the risk to overwrite the ckpts you might need. "
                "Set 'overwrite_output_dir' to true to confirm that you do want to overwrite the deprecated ckpts."
            )

        self.validate_megatron_args()

        if self.rng_types is None:
            self.rng_types = ["generator"]

        requires_backends(self, ["torch"])

        # TODO avoid being hard coded.
        self.load_model_func_kwargs["map_location"] = "on_device"

        self.distributed_state = None

        # TODO: temporarily
        if self.distributed_type == "megatron":
            if "distributed_timeout_minutes" in self.extra_configs:
                distributed_timeout_seconds = self.extra_configs["distributed_timeout_minutes"] * 60
                if self.ddp_timeout < distributed_timeout_seconds:
                    logger.warning(
                        f"AtorchTrainingArgs.ddp_timeout:{self.ddp_timeout}s will be overwritten by Megatron's args."
                        f"distributed_timeout_minutes:{self.extra_configs['distributed_timeout_minutes']}seconds."
                    )
                    self.ddp_timeout = distributed_timeout_seconds
            if "distributed_backend" in self.extra_configs:
                if self.extra_configs["distributed_backend"] != self.ddp_backend:
                    logger.warning(
                        f"AtorchTrainingArgs.distributed_backend:{self.ddp_backend} is not equal to Megatron's "
                        f"args.distributed_backend:{self.extra_configs['distributed_backend']}, and "
                        f"args.distributed_backend:{self.extra_configs['distributed_backend']} will be used as "
                        "torch's collection communication backend."
                    )
                    self.ddp_backend = self.extra_configs["distributed_backend"]

        # TODO: maybe setup device when init engine.
        self._setup_devices

        if self.profiler_type in ["hw", "hw_dp"] and not is_torch_npu_available():
            raise ValueError(
                f"profiler_type is set to {self.profiler_file_path}, but torch_npu is not available. "
                "Please check if torch_npu is installed."
            )

    def megatron_args(self) -> "MegatronArgs":
        if self._megatron_args is not None:
            return self._megatron_args
        else:
            megatron_dict = self.do_auto_mapping(COMMON_TO_MEGATRON_ARG_MAP)
            self._megatron_args = MegatronArgs(**megatron_dict)
            return self._megatron_args

    # TODO: Now called at Trainer.__init__()
    @cached_property
    def _setup_devices(self):
        # TODO: Improve setup devices.
        logger.info("PyTorch: setting up devices")

        self.distributed_state = AtorchAcceleratorState(self)

        device = torch.device("cuda", self.distributed_state.local_process_index)

        return device

    @property
    def device(self) -> torch.device:
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    def world_size(self):
        """
        The number of processes used in parallel.
        """
        return self.distributed_state.num_processes

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        return self.distributed_state.process_index

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        return self.distributed_state.local_process_index

    @property
    def is_main_process(self):
        """
        Whether or not this process is the global main process.
        """
        return self.distributed_state.is_main_process

    @property
    def is_local_main_process(self):
        """
        Whether or not this process is the local main process.
        """
        return self.distributed_state.is_local_main_process

    @property
    def use_distributed(self):
        """
        Whether configured for distributed training
        """
        return self.distributed_type != "no" and self.world_size > 1

    @property
    def dp_group_size(self):
        """
        The size of DP group.
        """
        return self.distributed_state.dp_group_size

    @property
    def global_train_batch_size(self) -> int:
        """
        The actual global batch size for training, along DP group.
        """
        return self.per_device_train_batch_size * self.dp_group_size

    @property
    def global_eval_batch_size(self) -> int:
        """
        The actual global batch size for evaluation, along DP group.
        """
        return self.per_device_eval_batch_size * self.dp_group_size

    def print(self):
        print("Atorch training args:")
        print(self.to_dict())

    @contextlib.contextmanager
    def main_process_first(self, local=True, desc="work"):
        """
        A context manager for torch distributed environment where on needs to do something on the main process, while
        blocking replicas, and when it's finished releasing the replicas.

        One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
        which upon completion saves a cached version of results and which then automatically gets loaded by the
        replicas.

        Args:
            local (`bool`, *optional*, defaults to `True`):
                if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
                rank 0 In multi-node environment with a shared filesystem you most likely will want to use
                `local=False` so that only the main process of the first node will do the processing. If however, the
                filesystem is not shared, then the main process of each node will need to do the processing, which is
                the default behavior.
            desc (`str`, *optional*, defaults to `"work"`):
                a work description to be used in debug logs

        """
        if is_torch_available() and self.world_size > 1:
            main_process_desc = "main local process" if local else "main process"
            assert self.distributed_state is not None, (
                "Please use a AtorchTrainingArgs object to call 'main_process_first()', "
                "like 'training_args.main_process_first'"
            )
            is_main_process = (
                self.distributed_state.is_local_main_process if local else self.distributed_state.is_main_process
            )

            try:
                if not is_main_process:
                    # tell all replicas to wait
                    logger.debug(f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}")

                    dist.barrier()
                yield
            finally:
                if is_main_process:
                    # the wait is over
                    logger.debug(f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas")
                    dist.barrier()
        else:
            yield


@dataclass
class MegatronArgs(DynamicDataClass):
    """
    Arguments for Megatron-LM to enable tensor, pipeline, data, sequence and expert parallelism.
    Also to enable selective activation recomputation and optimized fused kernels.
    """

    tensor_model_parallel_size: int = field(default=None, metadata={"help": "tensor parallelism degree."})

    pipeline_model_parallel_size: int = field(default=None, metadata={"help": "pipeline parallelism degree."})

    num_micro_batches: int = field(default=1, metadata={"help": "number of micro-batches."})

    consumed_samples: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Number of samples consumed in the same order as the dataloaders to `accelerator.prepare` call."
        },
    )

    # whether user use a pre-processed megatron dataset
    megatron_dataset_flag: bool = field(
        default=False,
        metadata={"help": "Whether the format of dataset follows Megatron-LM Indexed/Cached/MemoryMapped format."},
    )

    custom_megatron_dataloaders_provider_function: Optional[Callable] = field(
        default=None,
        metadata={
            "help": "Custom megatron train_valid_test dataloader provider function. Be aware, if this arg is set,"
            "custom_megatron_datasets_provider_function and data_sets parameter in atorch trainer will be ignored."
        },
    )

    custom_megatron_datasets_provider_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom megatron train_valid_test datasets provider function."},
    )

    # TODO check and clean up this arg
    is_train_batch_min: bool = field(
        default=True,
        metadata={"help": "If both train & eval dataloaders are specified, this will decide the micro_batch_size"},
    )

    # Custom train step args
    custom_train_step_class: Optional[Any] = field(default=None, metadata={"help": "Custom train step class."})
    custom_train_step_kwargs: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Custom train step kwargs."}
    )

    # Custom function to prepare megatron model.
    custom_model_provider_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom model provider function."},
    )
    custom_prepare_model_function: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom prepare model function."},
    )

    custom_tensorboard_record_calculate_fn: Optional[Callable] = field(
        default=None, metadata={"help": "function to calculate custom tensorboard record."}
    )

    # Megatron optimizer
    no_wd_decay_cond: Optional[Callable] = field(default=None, metadata={"help": "Condition to disable weight decay."})
    scale_lr_cond: Optional[Callable] = field(default=None, metadata={"help": "Condition to scale learning rate."})
    lr_mult: float = field(default=1.0, metadata={"help": "Learning rate multiplier."})

    # Megatron init arguments
    extra_args_provider: Optional[Callable] = field(
        default=None,
        metadata={"help": "Custom args, as a complement for Megatron-LM arguments."},
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@dataclass
class DeepSpeedArgs(AtorchTrainingArgs):
    """
    a placeholder
    """

    pass


@dataclass
class AtorchDPOTrainingArgs(AtorchTrainingArgs):
    """
    a placeholder for DPO trainer
    """

    pass
