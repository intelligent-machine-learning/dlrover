from atorch.auto.opt_lib.amp_optimization import AmpNativeOptimization, Fp8Optimization
from atorch.auto.opt_lib.checkpoint_optimization import CheckpointOptimization
from atorch.auto.opt_lib.ds_3d_parallel_optimization import DeepSpeed3DParallelOptimization
from atorch.auto.opt_lib.dynamo_optimization import NativeDynamoOptimization
from atorch.auto.opt_lib.half_optimization import HalfOptimization
from atorch.auto.opt_lib.mixed_parallel_optimization import MixedParallelOptimization
from atorch.auto.opt_lib.module_replace_optimization import ModuleReplaceOptimization
from atorch.auto.opt_lib.parallel_mode_optimization import ParallelModeOptimization
from atorch.auto.opt_lib.pipeline_parallel_optimization import PipelineParallelOptimization
from atorch.auto.opt_lib.sequence_parallel_optimization import SequenceParallelOptimization
from atorch.auto.opt_lib.tensor_parallel_optimization import TensorParallelOptimization
from atorch.auto.opt_lib.zero_optimization import FSDPOptimization, Zero1Optimization, Zero2Optimization
from atorch.common.log_utils import default_logger as logger

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
            Zero1Optimization,
            Zero2Optimization,
            FSDPOptimization,
            ParallelModeOptimization,
            AmpNativeOptimization,
            Fp8Optimization,
            TensorParallelOptimization,
            ModuleReplaceOptimization,
            CheckpointOptimization,
            NativeDynamoOptimization,
            PipelineParallelOptimization,
            MixedParallelOptimization,
            SequenceParallelOptimization,
            HalfOptimization,
            DeepSpeed3DParallelOptimization,
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
                logger.error(f"Invalid optimization method: {name}")
                break
        return valid
