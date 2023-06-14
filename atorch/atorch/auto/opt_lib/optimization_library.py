from atorch.auto.opt_lib.amp_optimization import AmpApexO1Optimization, AmpApexO2Optimization, AmpNativeOptimization
from atorch.auto.opt_lib.checkpoint_optimization import CheckpointOptimization
from atorch.auto.opt_lib.half_optimization import HalfOptimization
from atorch.auto.opt_lib.module_replace_optimization import ModuleReplaceOptimization
from atorch.auto.opt_lib.parallel_mode_optimization import ParallelModeOptimization
from atorch.auto.opt_lib.zero_optimization import FSDPOptimization, Zero1Optimization, Zero2Optimization

SEMIAUTO_STRATEGIES = ("tensor_parallel", "mixed_parallel", "pipeline_parallel")


class OptimizationLibrary(object):
    """
    Optimization library
    """

    def __init__(self):
        self.opts = {}
        self.group = {}
        self.register_optimizations()

    def __getitem__(self, name):
        if name not in self.opts:
            return None
        return self.opts[name]

    def register_opt(self, opt):
        self.opts[opt.name] = opt
        if opt.group in self.group:
            self.group[opt.group].append(opt.name)
        else:
            self.group[opt.group] = [opt.name]

    def register_optimizations(self):
        opt_list = [
            AmpApexO1Optimization,
            AmpApexO2Optimization,
            Zero1Optimization,
            Zero2Optimization,
            FSDPOptimization,
            ParallelModeOptimization,
            AmpNativeOptimization,
            ModuleReplaceOptimization,
            CheckpointOptimization,
            HalfOptimization,
        ]
        for opt in opt_list:
            opt_instance = opt()
            self.register_opt(opt_instance)

    def validate_strategy(self, strategy):
        valid = True
        for (name, config, tunable) in strategy:
            opt = self[name]
            if (
                opt is None
                or (opt.is_tunable is False and tunable is True)
                or (opt.is_tunable is True and tunable is False and config is None)
            ):
                valid = False
                break
        return valid
