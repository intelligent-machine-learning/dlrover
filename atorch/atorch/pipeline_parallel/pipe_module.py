from typing import Callable, Dict, List, Optional

from torch import nn

from atorch.pipeline_parallel.pipe_partition import (
    TieWeightInfo,
    partition_model_from_meta_model,
    partition_model_from_model_provider,
)


class PipeModuleConfig:
    def __init__(
        self,
        partition_method="default",
        module_list=None,
        model_config=None,
        total_layer_num=None,
        tie_weight_info=None,
        input_output_mapping=None,
        virtual_pp_size=None,
        accelerate_strategy=None,
        auto_ddp=True,
    ):
        self.partition_method = partition_method
        self.module_list = module_list
        self.model_config = model_config
        self.total_layer_num = total_layer_num
        self.tie_weight_info = tie_weight_info
        self.input_output_mapping = input_output_mapping
        self.virtual_pp_size = virtual_pp_size
        self.accelerate_strategy = accelerate_strategy
        self.auto_ddp = auto_ddp


class PipeModule(nn.Module):
    def __init__(
        self,
        modules: List[nn.Module],
        stage_ids: List[int],
        num_stages: int,
        input_output_mapping: Dict,
        loss_func: Optional[Callable] = None,
        meta_model: Optional[nn.Module] = None,
        tie_weight_info: Optional[TieWeightInfo] = None,
    ):
        """
        modules: list of nn.Module belongs to current pipe rank. The list length equals virtual pipeline size.
        stage_ids: list of in corresponding to stage indices of modules.
        input_output_mapping (optional): if not None, it is a dict, with "default" or stage_id (int) as key, and
            value is a 2-item tuple, the first item is activation_mapping, the second item is batch_mapping.
            If activation_mapping is None, the corresponding stage will use all activations received from previous
            stages in its forward inputs. If not None, it is a list of pair[name_in_forward, activation_idx],
            so assign this stage forward input parameter 'name_in_forward' with activaiton indexed
            by activation_idx.
            If batch_mapping is None, the corresponding stage will not use any batch data.
            If not None, it is a list of pair[name_in_forward, name_in_batch], so assign this stage
            forward parameter 'name_in_forward' with batch[name_in_batch].
        loss_func (optional): required by last stage, which takes last stage's output and batch data as input.
        meta_model (optional): if not None, a nn.Module, and modules are submodules in meta_model.
                    Modules are materialized and other submobules in meta_model are still on meta device.
        tie_weight_info (optional): tie weights info
        """
        nn.Module.__init__(self)
        assert len(modules) == len(stage_ids), "modules and pp_ranks should have same length"
        self.modules = modules
        self.stage_ids = stage_ids
        self.num_stages = num_stages
        self.loss_func = loss_func if num_stages - 1 in stage_ids else None
        self.meta_model = meta_model
        self.input_output_mapping = input_output_mapping
        self.tie_weights_info = tie_weight_info
        self.virtual_pp_size = len(stage_ids)
        self.current_chunk_id = 0
        if meta_model is None:
            for rank, m in zip(stage_ids, modules):
                self.register_module(str(rank), m)
                # todo: update tie_weights_info's weight name to add prefix.
        else:
            # exclude meta params/modules
            self._parameters = {}
            self._buffers = {}
            self._modules = {}
            for m in modules:
                self._parameters.update(m._parameters)
                self._buffers.update(m._buffers)
                self._modules.update(m._modules)
        self.set_current_chunk_id(0)  # assign hooks
        # add hooks for tie weight sync if tie exists.

    def forward(self, *args, **kargs):
        output = self.modules[self.current_chunk_id].forward(*args, **kargs)
        # also return loss func is last stage. Should we return loss directly?
        loss_func = self.loss_func if self.get_current_stage_idx() == self.num_stages - 1 else None
        return output, loss_func

    def set_current_chunk_id(self, index):
        self.current_chunk_id = index
        # TODO: also update forward/backward hooks

    def get_current_stage_idx(self):
        return self.stage_ids[self.current_chunk_id]

    def apply_stategy(self, strategy=None, auto_ddp=True):
        # TODO: apply strategy to modules, and ddp wrap if needed.
        pass

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

    def get_io_mapping(self, virtual_pp_rank=0):
        cur_stage = self.stage_ids[virtual_pp_rank]
        if cur_stage in self.input_output_mapping:
            mapping = self.input_output_mapping[cur_stage]
        else:
            mapping = self.input_output_mapping["default"]
        return mapping


def adjust_input_output_mapping(input_output_mapping, num_stages):
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


def make_pipe_module(meta_model=None, model_provider=None, loss_func=None, distributed_context=None, config=None):
    """
    Input params
    meta_model: Optional[nn.Module], meta model
    model_provider: Optional[Callable], model_provider func
    loss_func: Optional[Callable], loss function takes inputs and last stage outputs as input.
    distributed_context: Optional[DistributedContext], to get ranks/pgs.
    config: PipeModuleConfig, configs

    User must provide either meta_model or model_provider, but not both.
    Return a PipeModule instance.
    """
    assert (meta_model is None and model_provider is not None) or (
        meta_model is not None and model_provider is None
    ), "Provide either meta_model or model_provider, but not both"
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
        meta_model,
        tie_weight_info,
    )

    # step 4: apply streategy and ddp if needed.
    pipe_module.apply_stategy(strategy=config.accelerate_strategy, auto_ddp=config.auto_ddp)

    return pipe_module
