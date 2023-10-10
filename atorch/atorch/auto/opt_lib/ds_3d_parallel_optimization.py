import deepspeed
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
from transformers.modeling_utils import PreTrainedModel

from atorch.auto.opt_lib.optimization import Optimization
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import _DistributedContext as dc
from atorch.distributed.distributed import parallel_config, parallel_group_size
from atorch.modules.distributed_modules.randomizer import get_MDPRInstance, init_randomizer
from atorch.utils.ds_pipe_utils import PipeModuleFromRecordedMeta
from atorch.utils.manual_tp_utils import (
    TPInfo,
    hf_init_weights_custom_fn,
    tp_manual_shard_custom_fn,
    vocab_parallel_logit_helper,
)


class DeepSpeed3DParallelConfig:
    def __init__(
        self,
        base_seed=1234,
        tpinfo=None,
        custom_patcher=None,
        tie_first=True,
        logit_helper=None,
        ds_config=None,
        batch_fn=None,
    ):
        self.base_seed = base_seed

        # TPinfo
        self.tpinfo = tpinfo if tpinfo is not None else TPInfo()

        # PipeModuleFromRecordedMeta
        self.custom_patcher = custom_patcher if custom_patcher is not None else dict()
        self.tie_first = tie_first
        # logit helper
        if self.tpinfo.is_vocab_parallelled:
            if logit_helper is None:
                self.logit_helper = vocab_parallel_logit_helper
            else:
                logger.warning("Tensor parallel is using VocabParallelEmb, make sure lm_output copied to group")
        else:
            self.logit_helper = logit_helper

        # DeepSpeed config
        self.ds_config = ds_config  # dict() or path

        self.batch_fn = batch_fn if batch_fn is not None else lambda x: x


class DeepSpeed3DParallelOptimization(Optimization):
    """
    DeepSpeed 3D parallel optimization
    """

    def __init__(self):
        super().__init__("deepspeed_3d_parallel", "parallel", True)

    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, config, model_context

    def transform(self, model_context, config=DeepSpeed3DParallelConfig()):
        model_context.add_wrapper(
            "ds_3d_parallel", DeepSpeed3DParallelOptimization.apply_wrapper, wrapper_config=config, is_pre_wrapper=False
        )
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        assert isinstance(wrapper_config, DeepSpeed3DParallelConfig), (
            f"Invalid config for DeepSpeed3DParallelOptimization. "
            f"Should be DeepSpeed3DParallelConfig but get {wrapper_config}"
        )
        cfg = wrapper_config

        # get meta_model, optimizer, loss_fn
        model = model_context.model
        optim_func = model_context.optim_func
        optim_args = model_context.optim_args
        optim_param_func = model_context.optim_param_func
        loss_fn = model_context.loss_func  # vocab_parallel_cross_entropy if VocabParallelEmbedding

        # init randomizer and patch deepspeed checkpointing get_cuda_rng_tracker
        init_randomizer(cfg.base_seed)
        deepspeed.checkpointing.get_cuda_rng_tracker = get_MDPRInstance

        # huggingface _init_weights custom fn
        if isinstance(model, PreTrainedModel):
            hf_init_weights_custom_fn(model)

        # tensor parallel custom fn
        tp_manual_shard_custom_fn(model, cfg.tpinfo)

        # deepspeed topology
        assert dc.INITIALIZED, "_DistributedContext not initialized."
        assert tuple(group[0] for group in parallel_config()[0]) == (
            "tensor",
            "data",
            "pipeline",
        ), f"Invalid parallel config, must be in [tensor, data, pipeline] order but get {parallel_config()}."
        topo = PipeModelDataParallelTopology(
            num_pp=parallel_group_size("pipeline"),
            num_mp=parallel_group_size("tensor"),
            num_dp=parallel_group_size("data"),
        )

        # build pipeline module
        pipeline_module = PipeModuleFromRecordedMeta(
            model,
            custom_patcher=cfg.custom_patcher,
            tie_first=cfg.tie_first,
            logit_helper=cfg.logit_helper,
            topology=topo,
            loss_fn=loss_fn,
        )

        # construct optimizer
        optimizer = optim_func(optim_param_func(pipeline_module), **optim_args)

        # deepspeed pipeline engine
        model, optimizer, _, _ = deepspeed.initialize(model=pipeline_module, optimizer=optimizer, config=cfg.ds_config)
        model.set_batch_fn(cfg.batch_fn)  # batch_fn for deepspeed pipeline (inputs,), (labels,) compat

        model_context.model = model
        model_context.optim = optimizer
        # model_context.loss_fn will not be used, use model.train_batch / eval_batch instead
        return model_context
