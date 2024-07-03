class TieWeightInfo:
    def __init__(self):
        self.tie_info = []

    def add(self, info):
        # info is a list of weights tied together.
        # info: Union[List[str], List[Tuple(int, str)]]]
        # either weight name, or (stage_id, weight_name)
        # weights in the same list are tied.
        self.tie_info.append(info)

    def num(self):
        return len(self.tie_info)

    def __getitem__(self, index):
        return self.tie_info[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __delitem__(self, index):
        del self.data[index]


def partition_model_from_model_provider(model_provider, distributed_context, config):
    modules = None
    stage_ids = None
    tie_weight_info = None

    pp_size = (
        distributed_context.parallel_group_size("pipe")
        if distributed_context.parallel_group_size("pipe") is not None
        else 1
    )
    pp_rank = distributed_context.parallel_rank("pipe") if distributed_context.parallel_rank("pipe") is not None else 0
    virtual_pp_size = config.virtual_pp_size if config.virtual_pp_size is not None else 1
    total_stages = pp_size * virtual_pp_size

    stage_ids = [pp_rank + i * pp_size for i in range(virtual_pp_size)]

    assert (
        config.total_layer_num >= total_stages
    ), f"Total layer num({config.total_layer_num}) should greater than or equal to total stage num({total_stages})."

    per_stage_layer_num = config.total_layer_num // total_stages
    extra_layer_stage_num = config.total_layer_num % total_stages

    modules = []

    for stage_id in stage_ids:
        if extra_layer_stage_num == 0:
            layer_num = per_stage_layer_num
        elif stage_id > 0 and stage_id <= extra_layer_stage_num:
            layer_num = per_stage_layer_num + 1
        pre_process = stage_id == 0
        post_process = stage_id == total_stages - 1
        module = model_provider(
            model_config=config.model_config, layer_num=layer_num, pre_process=pre_process, post_process=post_process
        )
        modules.append(module)

    # TODO: get tie_weight_info
    tie_weight_info = None

    return modules, stage_ids, tie_weight_info


def partition_model_from_meta_model(meta_model, distributed_context, config):
    assert "Not implemented yet."
    return None, None, None
