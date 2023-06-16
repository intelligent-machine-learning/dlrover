import copy

import numpy
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from atorch.auto.model_context import ModelContext
from atorch.auto.opt_lib.module_replace_optimization import REPLACEMENT_PAIRS
from atorch.common.constants import AnalyserConstants
from atorch.common.log_utils import default_logger as logger


class Analyser(object):
    def __init__(self):
        """
        Register available analysis methods
        """
        self._methods = {}
        self._register_methods()

    def analyse(self, model_context, methods):
        """
        analysis model_context using methods.
        Args:
            model_context(ModelContext): model context
            methods([str]): a list of string of analysis methods
        Returns(dict): analyze result
        """
        results = {}

        # call analyze method
        for method in methods:
            if method not in self._methods:
                raise NotImplementedError(f"Method: {method} is not supported in Analyser!")
            else:
                res = self._methods[method](model_context)
                results[method] = res

        return results

    def _register_methods(self):
        self._methods[AnalyserConstants.ANALYSE_BASIC] = self._analyse_basic
        self._methods[AnalyserConstants.ANALYSE_TRANSFORMER] = self._analyse_transformer
        self._methods[AnalyserConstants.ANALYSE_DYNAMIC] = self._analyse_dynamic

    def _analyse_basic(self, model_context):
        logger.info("Analyse model basic info...")
        res = {}

        model = model_context.model
        if not isinstance(model, nn.Module):
            raise ValueError(f"model: {model}, must be a instance of {nn.Module}")
        res[AnalyserConstants.MODEL_PARAMS_NUM] = self._get_model_params_num(model)
        res[AnalyserConstants.MODEL_PARAMS_MB] = self._get_model_params_mb(model)
        res[AnalyserConstants.HAS_MODULE_FOR_REPLACE] = self._has_module_for_replace(model)

        optimizer = model_context.create_optim()
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(f"optimizer: {optimizer}, must be a instance of {torch.optim.Optimizer}")
        res[AnalyserConstants.OPTIMIZER_TYPE] = self._get_optimizer_type(optimizer)
        res[AnalyserConstants.SUBMODULE_TYPES] = self._get_submodule_types(model)
        res[AnalyserConstants.OPT_CONFIG_SUBMODULE_NAMES] = self._get_opt_config_submodule_names(model)
        return res

    def _get_model_params_num(self, model):
        """
        Get params number of the model
        Args:
            model: model(required)
        Returns:
            the number of parameters of model
        """
        num = 0
        for _, param in model.named_parameters():
            num += param.nelement()
        return num

    def _get_model_params_mb(self, model):
        """
        Get params size of the model
        Args:
            model: model(required)
        Returns:
            size(MB) of parameters in model
        """
        total_size = 0
        for _, param in model.named_parameters():
            total_size += param.nelement() * param.element_size() / 1024 / 1024
        return total_size

    def _has_module_for_replace(self, model):
        """
        Check if model has modules that can be optimized by replacement
        Args:
            model: model(required)
        Returns:
            True if has module for replace, False otherwise.
        """
        module_list = []
        for module, _, _, _ in REPLACEMENT_PAIRS.values():
            module_list.append(module)
        module_list = tuple(module_list)
        for _, m in model.named_modules():
            if isinstance(m, module_list):
                return True
        return False

    def _analyse_dynamic(self, model_context):
        logger.info("Analyse model dynamic info...")
        res = {}
        local_model_context = copy.deepcopy(model_context)
        dataloader = local_model_context.create_dataloader({"batch_size": 1, "shuffle": True})
        model = local_model_context.model
        optimizer = local_model_context.create_optim()
        loss_func = local_model_context.loss_func
        prepare_input = local_model_context.prepare_input
        model_input_format = local_model_context.model_input_format
        res[AnalyserConstants.DATA_SIZE], res[AnalyserConstants.FIXED_DATA_SIZE] = self._get_single_sample_size(
            dataloader
        )
        res[AnalyserConstants.MODEL_FLOPS_AND_DYNAMIC_MEMORY_MB] = self._get_model_flops_and_dynamic_memory_mb(
            model, optimizer, dataloader, loss_func, prepare_input, model_input_format
        )
        res[AnalyserConstants.OPTIMIZER_STATE_NUM_AND_MEMORY_MB] = self._get_optimizer_state_num_and_memory_mb(
            optimizer
        )
        del local_model_context

        return res

    def _get_single_sample_size(self, dataloader):
        """
        Get single sample size
        Args:
            dataloader: torch.utils.data.DataLoader
        Returns:
            sample size(int), fixed_data_size(bool)
        """
        sample_size = 0
        fixed_data_size = True
        for i in range(20):
            single_sample = next(iter(dataloader))
            size = self._get_data_size_recursively(single_sample)
            if sample_size > 0 and sample_size != size:
                logger.warning("The sample size is various.")
                fixed_data_size = False
                break
            else:
                sample_size = size
        return sample_size, fixed_data_size

    def _get_data_size_recursively(self, data):
        if isinstance(data, torch.Tensor):
            return data.numel()
        elif isinstance(data, dict):
            return sum([self._get_data_size_recursively(data[key]) for key in data])
        elif isinstance(data, (list, tuple)):  # namedtuple included
            return sum(self._get_data_size_recursively(d) for d in data)
        elif isinstance(data, numpy.ndarray):
            return data.size
        elif isinstance(data, str):
            return len(data)
        elif isinstance(data, int):
            return 1
        elif isinstance(data, float):
            return 1
        else:
            raise TypeError("unsupported data type %s" % type(data))

    def _get_model_flops_and_dynamic_memory_mb(
        self, model, optimizer, dataloader, loss_func, prepare_input, model_input_format
    ):
        res = {}
        if torch.cuda.is_available():
            activities = [ProfilerActivity.CUDA]
            model.cuda()
            device = "cuda"
        else:
            activities = [ProfilerActivity.CPU]
            device = "cpu"
        model.train()
        with profile(activities=activities, profile_memory=True, with_flops=True) as prof:
            for batch, data in enumerate(dataloader):
                data = prepare_input(data, device)
                if model_input_format == "unpack_dict":
                    output = model(**data)
                elif model_input_format == "unpack_sequence":
                    output = model(*data)
                else:
                    output = model(data)
                loss = ModelContext.get_loss_from_loss_func_output(loss_func(data, output))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch >= 20:
                    break
        total_memory = 0
        total_flops = 0
        for event in prof.events():
            if torch.cuda.is_available():
                total_memory += event.cuda_memory_usage
            else:
                total_memory += event.cpu_memory_usage
            if event.flops:
                total_flops += event.flops
        res[AnalyserConstants.DYNAMIC_MEMORY_MB] = total_memory / 20 / 1024 / 1024
        res[AnalyserConstants.MODEL_FLOPS] = total_flops / 20

        return res

    def _get_optimizer_state_num_and_memory_mb(self, optimizer):
        res = {}
        state_num = 0
        for state in optimizer.state_dict()["state"].values():
            for state_name in state:
                if state[state_name] is not None and isinstance(state[state_name], torch.Tensor):
                    state_num += state[state_name].numel()
        res[AnalyserConstants.OPTIMIZER_STATE_NUM] = state_num
        res[AnalyserConstants.OPTIMIZER_STATE_MEMORY_MB] = state_num * 4 / 1024 / 1024
        return res

    def _analyse_transformer(self, model_context):
        """
        Get transformer structure of the model
        Args:
            model_context(ModelContext): model context
        Returns:
            dict(class:int): key is transformer structure class, val is number of the structure
        """
        from transformers.models.bert.modeling_bert import (
            BertAttention,
            BertEmbeddings,
            BertIntermediate,
            BertLayer,
            BertOutput,
            BertPooler,
            BertSelfAttention,
            BertSelfOutput,
        )

        supported_structure = (
            BertLayer,
            BertAttention,
            BertEmbeddings,
            BertSelfAttention,
            BertSelfOutput,
            BertPooler,
            BertIntermediate,
            BertOutput,
        )
        logger.info("Analyse transformer structure...")
        res = {}
        model = model_context.model
        for _, module in model.named_modules():
            if isinstance(module, supported_structure):
                if module.__class__.__name__ not in res:
                    res[module.__class__.__name__] = 0
                res[module.__class__.__name__] += 1

        return res

    def _get_optimizer_type(self, optimizer):
        """
        Get optimizer type
        Args:
            optimizer: optimizer(required)
        Returns:
            str: optimizer class name
        """
        return optimizer.__class__.__name__

    def _get_submodule_types(self, model, depth=None):
        """This method gets all submodule cls, depth controls how deep we
        track into the root module. Module cls returned by this method can be candidates
        for CheckpointOptimization's wrapper_config
        """
        # do a naive BFS and controls the depth of traversal
        node_depth = {}
        queue = [model]
        node_depth[model] = 0
        submodule_types = set()
        while queue:
            root_model = queue.pop(0)
            for child_mod in root_model.children():
                if child_mod not in node_depth:
                    if depth is None or node_depth[root_model] <= depth:
                        node_depth[child_mod] = node_depth[root_model] + 1
                        queue.append(child_mod)
                        submodule_types.add(type(child_mod))

        return tuple(submodule_types)

    def _get_opt_config_submodule_names(self, model, depth=None):
        """This method gets all submodule cls and return the potential submodule names for Zero2Optimization,
        FSDPOptimization, and CheckpointOptimization
        depth: controls how deep we track into the root module.
        """
        # do a naive BFS and controls the depth of traversal
        opt_config_submodule_names_list = [
            "GPT2Block",
            "CLIPAttention",
            "CLIPMLP",
            "CLIPAttentionFA",
            "GPTNeoXAttention",
            "GPTNeoXMLP",
            "ResidualAttentionBlock",
            "BertLayer",
        ]
        submodule_names_set = set()
        for m in model.modules():
            module_name = type(m).__name__
            if module_name in opt_config_submodule_names_list:
                submodule_names_set.add(module_name)

        return tuple(submodule_names_set)


_ANALYSER = None


def get_analyser():
    global _ANALYSER
    if _ANALYSER is None:
        _ANALYSER = Analyser()
    return _ANALYSER
