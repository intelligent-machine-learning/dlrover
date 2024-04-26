from functools import partial

import atorch
from atorch.auto.opt_lib.optimization import Optimization
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import create_sequence_parallel_group, destroy_sequence_parallel_group


class SequenceParallelOptimization(Optimization):
    """SequenceParallelOptimization implements sequence parallel(SP) for Transformer model similar to DS-Ulysses,
    which requires that num_heads % sp_size == 0 and sequence_len % sp_size == 0. Thus, for Attention module,
    it is model parallel on head dimension. For other parts, it is data parallel on sequence dimension.
    Unlike DS-ULysses, atorch's sequence parallel group is independent to data parallel group. For example,
    if data parallel group size is 32, and sequence parallel group size is 8, then every 32 nodes will have
    1 DP group and 4 SP groups. DP's allreduce/reduce_scatter op would responsible for combine all the gradients
    updated by DP and SP.
    """

    def __init__(self):
        super().__init__("sequence_parallel", "parallel", False)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        # config is a dict contains:
        #     "sp_size": int
        #     "module": Optional, Oneof(None, class, list/tuple of classes). Default None.
        #          if None, model_context.model supports set_sp interface.
        #          if class or list/tuple of classes, any modules with such classes support set_sp interface.
        #     "set_sp_func_name": Optional, set_sp func name in module, default "set_sp"
        #     "batch_sp_processing_fn": optional, func to split batch data into sub-sequence,
        #          with inerface as func(batch, sp_size, sp_rank).
        #          If not provided, user is responsible for split sequence in input data batch.
        # model or modules satisfying isinstance(module, config["module"]) will be called
        # module.set_sp_func_name(sp_size, sp_rank, sp_group).

        # 1. Create parallel group for sp
        create_sequence_parallel_group(config["sp_size"])

        # 2. Add wrapper
        model_context.add_wrapper(
            "sequence_parallel", SequenceParallelOptimization.apply_wrapper, wrapper_config=config, is_pre_wrapper=True
        )
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        sp_size = wrapper_config["sp_size"]
        sp_rank = atorch.distributed.get_sequence_parallel_rank()
        sp_group = atorch.distributed.get_sequence_parallel_group()
        sp_func_name = wrapper_config["set_sp_func_name"] if "set_sp_func_name" in wrapper_config else "set_sp"

        # 1. call sp_func_name for model/modules
        def _call_set_sp(module):
            if not hasattr(module, sp_func_name):
                logger.error(f"Module {name} does not have set_sp func with name {sp_func_name}")
            sp_func = getattr(module, sp_func_name)
            sp_func(sp_size, sp_rank, sp_group)

        found_count = 0
        if "module" not in wrapper_config or wrapper_config["module"] is None:
            _call_set_sp(model_context.model)
            found_count = 1
        else:
            if isinstance(wrapper_config["module"], (tuple, list)):
                module_types = tuple(wrapper_config["module"])
            else:
                module_types = wrapper_config["module"]
            for name, module in model_context.model.named_modules():
                if isinstance(module, module_types):
                    found_count += 1
                    _call_set_sp(module)
            if found_count == 0:
                logger.error("No modules found for sequence parallel!")
        logger.info(f"Setup sp with {found_count} {sp_func_name} calls.")

        # 2. add wrapper_config["batch_sp_processing_fn"] in prepare_input if exists.
        def _sp_process_func(data, device, sp_func, sp_size, sp_rank, ori_func):
            data = ori_func(data, device)
            return sp_func(data, sp_size, sp_rank)

        if (
            "batch_sp_processing_fn" in wrapper_config
            and wrapper_config["batch_sp_processing_fn"] is not None
            and model_context.prepare_input is not None
        ):
            model_context.prepare_input = partial(
                _sp_process_func,
                sp_func=wrapper_config["batch_sp_processing_fn"],
                sp_size=sp_size,
                sp_rank=sp_rank,
                ori_func=model_context.prepare_input,
            )

        return model_context

    @staticmethod
    def reset(config):
        """Delete sp parallel group."""
        destroy_sequence_parallel_group()
