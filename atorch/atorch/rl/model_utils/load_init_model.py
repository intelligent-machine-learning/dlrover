import importlib
import os

import deepspeed
import torch
import transformers  # noqa: F401
from torch import nn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from atorch.common.log_utils import default_logger as logger
from atorch.utils.import_util import import_module_from_py_file
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload


def get_optimizer(optimizer_name):
    if optimizer_name == "torch.optim.adam":
        return torch.optim.Adam
    elif optimizer_name == "torch.optim.adamw":
        return torch.optim.AdamW
    else:
        raise Exception("unsupport optimizer")


def get_scheduler_class(scheduler_name):
    cls = None
    if scheduler_name == "cosine_warmup":
        cls = get_cosine_schedule_with_warmup
    else:
        raise Exception("unsupport scheduler")
    return cls


def get_optimizer_class_and_kwargs(model: str, config):
    optimizer_cls, kwargs = None, None
    if model not in ["ref_model", "reward_model", "cost_model"]:
        name = config.optimizer.name
        kwargs = config.optimizer.kwargs
        optimizer_cls = get_optimizer(name)
    return optimizer_cls, kwargs


def load_transformers_pretrained_model(model_cls, model_dir, lazy_load, peft_config=None, **model_params):
    if model_cls.startswith("transformers"):
        # load hugging face model by function which
        # has from_pretrained method such as transformers.AutoModelForSeq2SeqLM
        model_cls = eval(model_cls)
        assert hasattr(model_cls, "from_pretrained"), "model doesn't from_pretrained method"
        from_pretrained = getattr(model_cls, "from_pretrained")
        if lazy_load:
            # TODO: test with loading multity model cases
            with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
                model = from_pretrained(model_dir, trust_remote_code=True)
        else:
            model = from_pretrained(model_dir, trust_remote_code=True)
    else:
        # user defined pretrained model, for exapmple
        # benchmarks.glm_rlhf.reward_model.reward_model.RewardModel
        # we need to import benchmarks.glm_rlhf.reward_model.reward_model module
        # and get RewardModel
        module = model_cls.split(".")[-1]
        model_class_path = model_cls.replace(module, "").strip(".")
        model_module = importlib.import_module(model_class_path)
        model_cls = getattr(model_module, module)
        hasattr(model_cls, "from_pretrained")
        from_pretrained = getattr(model_cls, "from_pretrained")
        if False:
            # TODO: test with loading multity model cases
            with init_empty_weights_with_disk_offload(ignore_tie_weights=False):
                model = from_pretrained(model_dir, **model_params)
        else:
            # only work for modeling definition in antnlp/solutions/anllm/examples/rlhf/rl
            if peft_config is None:
                model = from_pretrained(model_dir, **model_params)
            else:
                model = from_pretrained(model_dir, peft_config=peft_config, **model_params)
    return model


def initialize_model(model_cls, model_path, lazy_load, **model_params):
    model_module = import_module_from_py_file(model_path)
    module = getattr(model_module, model_cls)
    if not lazy_load:
        model = module(**model_params)
    else:
        # to be developed
        pass
    return model


def init_model(config):
    # for local debug mode model initialization
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = config.model_path
    model_cls = config.model_cls
    model_params = config.model_params
    peft_config = config.peft_config
    lazy_load = config.lazy_load
    if os.path.isdir(model_path):
        # for hugging face model, xxx.from_pretrained is used to load model
        # and model_path is dir
        model = load_transformers_pretrained_model(
            model_cls, model_path, lazy_load, **model_params, peft_config=peft_config
        )
    else:
        # model_path is python file path,
        # the module is imported and class is initialized
        model = initialize_model(model_cls, model_path, lazy_load, **model_params)
    # model.to(device)
    return model


def init_tokenizer(config):
    tokenizer_path = config.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tokenizer


def load_ds_model_with_zero3_partition(
    model_class,
    model_config_class,
    model_config_folder=None,
    checkpoint_path=None,
    ds_config_dict_or_path=None,
    prefix="",
):
    """
    Allocate a partitioned module, initialize its weight randomly
    Load checkpoint only on rank-0 to avoid oom and broadcast to other ranks.
    For models bigger than the memory of a single GPU, this method is required.
    """
    if checkpoint_path is None:
        logger.info("checkpoint path is None and tring to load pytorch_model.bin in {}".format(model_config_folder))
        checkpoint_path = os.path.join(model_config_folder, "pytorch_model.bin")
        assert os.path.exists(checkpoint_path), "{} doesn't exist".format(checkpoint_path)

    with deepspeed.zero.Init(config_dict_or_path=ds_config_dict_or_path):
        configuration = model_config_class.from_pretrained(model_config_folder, trust_remote_code=True)
        model = model_class(configuration)
    if deepspeed.comm.get_rank() == 0:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        state_dict = None

    def load(module: nn.Module, prefix=prefix):
        # because zero3 puts placeholders in model params, this context
        # manager gathers (unpartitions) the params of the current layer, then loads from
        # the state dict and then re-partitions them again
        with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
            if deepspeed.comm.get_rank() == 0:
                module._load_from_state_dict(state_dict, prefix, {}, False, [], [], [])
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix=prefix)
    torch.cuda.empty_cache()
    return model
