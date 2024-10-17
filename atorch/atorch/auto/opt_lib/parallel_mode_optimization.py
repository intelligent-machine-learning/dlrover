import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from atorch.auto.opt_lib.optimization import Optimization
from atorch.common.util_func import set_sync_bn_pg
from atorch.distributed.distributed import local_rank, parallel_group, parallel_group_size
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device


class ParallelModeOptimization(Optimization):
    def __init__(self):
        name = "parallel_mode"
        group = "parallel_mode"
        is_tunable = False
        super().__init__(name, group, is_tunable)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        model_context.parallel_mode_config = config
        for (name, size) in config[0]:
            if name == "data" and size > 1:
                set_sync_bn_pg(model_context.model, parallel_group("data"))
                model_context.add_wrapper("ddp", ParallelModeOptimization.apply_wrapper, "data", is_pre_wrapper=False)
                break
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config):
        if wrapper_name == "ddp":
            # move model to gpu device if needed.
            if torch.cuda.is_available():
                model_device = next(model_context.model.parameters()).device
                device = torch.device(type="cuda", index=local_rank())
                if model_device != device:
                    materialize_modules_to_device(model_context.model, device)
            model_context.model = DDP(
                model_context.model,
                process_group=parallel_group(wrapper_config),
                find_unused_parameters=model_context.find_unused_parameters,
            )


class AutoDDPOptimization(Optimization):
    def __init__(self):
        name = "auto_ddp"
        group = "parallel_mode"
        super().__init__(name, group, is_tunable=False, is_distributed=True)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=None):
        dp_size = parallel_group_size("data")
        if dp_size is not None and dp_size > 1:
            model_context.add_wrapper("ddp", ParallelModeOptimization.apply_wrapper, "data", is_pre_wrapper=False)
        return model_context
