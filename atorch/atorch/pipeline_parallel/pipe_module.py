from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import torch
from torch import nn

import atorch
from atorch.auto import auto_accelerate
from atorch.distributed import distributed as _distributed_context
from atorch.distributed.distributed import parallel_group
from atorch.pipeline_parallel.pipe_partition import (
    TieWeightInfo,
    partition_model_from_meta_model,
    partition_model_from_model_provider,
)
from atorch.pipeline_parallel.pipe_stage import PipeStage


class PipeModuleConfig:
    def __init__(
        self,
        partition_method="default",
        manual_stage_partition=None,
        model_config=None,
        total_layer_num=None,
        tie_weight_info=None,
        input_output_mapping=None,
        virtual_pp_size=None,
        auto_ddp=True,
        sche_name="Schedule1F1B",
        n_microbatches=1,
    ):
        self.partition_method = partition_method
        self.manual_stage_partition = manual_stage_partition
        self.model_config = model_config
        self.total_layer_num = total_layer_num
        self.tie_weight_info = tie_weight_info
        self.input_output_mapping = input_output_mapping
        self.virtual_pp_size = virtual_pp_size
        self.auto_ddp = auto_ddp
        self.sche_name = sche_name
        self.n_microbatches = n_microbatches


class PipeModule(nn.Module):
    def __init__(
        self,
        modules: List[nn.Module],
        stage_ids: List[int],
        num_stages: int,
        input_output_mapping: Dict,
        loss_func: Optional[Callable] = None,
        tie_weight_info: Optional[TieWeightInfo] = None,
        strategy: Optional[Any] = None,
        auto_ddp: bool = True,
        device=None,
    ):
        """
        Args:
            modules: list of nn.Module belongs to current pipe rank. The list length equals virtual pipeline size.
            stage_ids: list of in corresponding to stage indices of modules.
            input_output_mapping (optional): if not None, it is a dict, with "default" or stage_id (int) as key, and
                value is a 2-item tuple, the first item is activation_mapping, the second item is batch_mapping.
                If activation_mapping is None, the corresponding stage will use all activations received from previous
                stages in its forward inputs. If not None, it is a list of pair[name_in_forward, activation_idx],
                so assign this stage forward input parameter 'name_in_forward' with activaiton indexed
                by activation_idx.
                If batch_mapping is None, the corresponding stage will not use any batch data.
                If not None, it is a list of pair[name_in_forward, name_in_batch or index_in_batch],
                so assign this stage.
                forward parameter 'name_in_forward' with batch[name_in_batch].
            loss_func (optional): takes last stage's output and batch data as input.
            tie_weight_info (optional): tie weights info
            strategy: if not None, a strategy for all modules to apply,
                or tuple of strategies for corresponding modules.
            auto_ddp: if apply ddp automatically if "data" pg exists.
        """
        nn.Module.__init__(self)
        assert len(modules) == len(stage_ids), "modules and pp_ranks should have same length"
        self.modules = modules
        self.stage_ids = stage_ids
        self.num_stages = num_stages
        self.ori_loss_func = loss_func
        self.loss_func = loss_func if num_stages - 1 in stage_ids else None
        self.input_output_mapping = input_output_mapping
        self.tie_weights_info = tie_weight_info
        self.virtual_pp_size = len(stage_ids)
        self.current_chunk_id = 0
        self.strategy = strategy
        self.auto_ddp = auto_ddp

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device(type="cuda", index=atorch.local_rank())
        else:
            self.device = "cpu"
        self.pg_group = parallel_group("pipe")

        self.apply_strategy_if_needed()

        self._parameters = {}
        self._buffers = {}
        self._modules = {}
        for m in modules:
            self._parameters.update(m._parameters)
            self._buffers.update(m._buffers)
            self._modules.update(m._modules)
        # TODO: add hooks for tie weight sync if tie exists.

        self.stages = self.create_stages()

    def apply_strategy_if_needed(self):
        if self.strategy is None and not self.auto_ddp:
            return

        individual_strategy = isinstance(self.strategy, tuple)
        new_modules = []
        for idx, module in enumerate(self.modules):
            strategy = self.strategy[idx] if individual_strategy else self.strategy
            if strategy is None:
                cur_strategy = []
            else:
                cur_strategy = deepcopy(strategy)
            if self.auto_ddp:
                cur_strategy.append("auto_ddp")
            status, result, _ = auto_accelerate(
                module,
                loss_func=self.ori_loss_func if self.stage_ids[idx] == self.num_stages - 1 else None,
                load_strategy=cur_strategy,
            )
            assert status
            if result.loss_func is not None:
                self.loss_func = result.loss_func
            new_modules.append(result.model)
        self.modules = new_modules

    def create_stages(self):
        stages = []
        for idx, module in zip(self.stage_ids, self.modules):
            io_mapping = self.get_io_mapping(idx)
            stage = PipeStage(module, idx, self.num_stages, self.device, io_mapping, self.pg_group)
            stages.append(stage)
        return stages

    def requires_batch_input(self):
        """
        Check whether current chunk needs batch data as input.
        """
        for stage_id in self.stage_ids:
            if stage_id in self.input_output_mapping:
                mapping = self.input_output_mapping[stage_id][1]
            else:
                mapping = self.input_output_mapping["default"][1]
            if mapping is not None and len(mapping) > 0:
                return True

        return False

    def get_io_mapping(self, stage_idx=0):
        if stage_idx in self.input_output_mapping:
            mapping = self.input_output_mapping[stage_idx]
        else:
            mapping = self.input_output_mapping["default"]
        return mapping


def adjust_input_output_mapping(input_output_mapping, num_stages):
    input_output_mapping = deepcopy(input_output_mapping)
    # if only one stage, combine all batch mapping, and ignore activation mapping
    if num_stages == 1:
        mapping = {}
        for _, batch_mapping in input_output_mapping.values():
            if batch_mapping is not None:
                for fn, bn in batch_mapping:
                    if fn not in mapping:
                        mapping[fn] = bn
                    else:
                        assert mapping[fn] == bn, f"Conflict batch data key {bn} and {mapping[fn]} for input {fn}"

        return {"default": (None, list(mapping.items()) if len(mapping) > 0 else None)}

    # add default if not exist
    if "default" not in input_output_mapping:
        input_output_mapping["default"] = (None, None)

    # adjust negative stage id to positive according to num_stages
    adjust_ids = [stage_id for stage_id in input_output_mapping if isinstance(stage_id, int) and stage_id < 0]
    for id in adjust_ids:
        mapping = input_output_mapping[id]
        new_id = id % num_stages
        del input_output_mapping[id]
        input_output_mapping[new_id] = mapping
    # check valid keys
    for stage_id in input_output_mapping:
        assert (
            isinstance(stage_id, int) and stage_id < num_stages
        ) or stage_id == "default", "mapping key must be either an int and smaller than num_stages or 'default'"
    return input_output_mapping


def make_pipe_module(
    meta_model=None, model_provider=None, loss_func=None, strategy=None, distributed_context=None, config=None
):
    """
    Input params
    meta_model: Optional[nn.Module], meta model
    model_provider: Optional[Callable], model_provider func
    loss_func: Optional[Callable], loss function takes inputs and last stage outputs as input.
    strategy: Optional, a strategy or list of strategies for modules
    distributed_context: Optional[DistributedContext], to get ranks/pgs.
    config: PipeModuleConfig, configs

    User must provide either meta_model or model_provider, but not both.
    Return a PipeModule instance.
    """
    assert (meta_model is None and model_provider is not None) or (
        meta_model is not None and model_provider is None
    ), "Provide either meta_model or model_provider, but not both"

    if distributed_context is None:
        distributed_context = _distributed_context
    # Step 1: partition model into model chunks
    if model_provider is not None:
        modules, stage_ids, tie_weight_info = partition_model_from_model_provider(
            model_provider, distributed_context, config
        )
    else:
        modules, stage_ids, tie_weight_info = partition_model_from_meta_model(meta_model, distributed_context, config)

    # Step 2: tie weights if exists (TODO)

    # Step 3: create PipeModule
    virtual_pp_size = config.virtual_pp_size if config.virtual_pp_size is not None else 1
    pp_size = (
        distributed_context.parallel_group_size("pipe") if distributed_context.parallel_rank("pipe") is not None else 1
    )
    num_stages = pp_size * virtual_pp_size
    input_output_mapping = adjust_input_output_mapping(config.input_output_mapping, num_stages)

    pipe_module = PipeModule(
        modules,
        stage_ids,
        num_stages,
        input_output_mapping,
        loss_func,
        tie_weight_info,
        strategy=strategy,
        auto_ddp=config.auto_ddp,
    )

    return pipe_module
