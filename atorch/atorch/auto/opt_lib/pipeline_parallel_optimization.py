import collections
import copy
import traceback
import types

import torch
from torch.fx.graph_module import GraphModule

from atorch.auto.auto_accelerate_context import AutoAccelerateContext
from atorch.auto.opt_lib.module_replace_optimization import (
    _enable_flash_attn_by_attr,
    _get_default_replace_config,
    _replace_by_config,
)
from atorch.auto.opt_lib.optimization import DistributedGraphMixin, Optimization
from atorch.auto.opt_lib.shard_planners import BaseStagePlanner
from atorch.auto.opt_lib.utils import insert_split_point
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import local_rank, parallel_group_and_ranks, rank
from atorch.modules.distributed_modules.compilers import pippy_compiler
from atorch.modules.distributed_modules.modules_registry import _SHARDABLE_OPERATORS

try:
    from pippy.IR import MultiUseParameterConfig, pipe_split
except ImportError:
    MultiUseParameterConfig, pipe_split = None, None


def _check_pipe_split_inserted(gm):
    for node in gm.graph.nodes:
        if node.target == pipe_split:
            return True

    return False


def _backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    # No-op backward for pipe mode, disable customized loss.backward
    pass


# original loss_func is wrapped inside pipe_model
# pipe_model has only one output (that is loss)
# hacks backward method
def pipe_loss_func(data, output, amp=False):
    if output is None:
        output = torch.zeros((0,))
    if isinstance(output, collections.abc.Sequence):
        output[0].backward = types.MethodType(_backward, output[0])
    else:
        output.backward = types.MethodType(_backward, output)
    return output


# FIXME Implement more strategies, including DP/KL
class PipelineParallelOptimization(Optimization, DistributedGraphMixin):
    """Pipeline parallel optimization.
    This is a STANDALONE implementation, which means mixed parallel cannot be implemented as
    a stack of PipelineParallelOptimization and TensorParallelOptimization.

    Refer to MixedParallelOptimization for 3D parallel.
    """

    def __init__(self, shard_planner="equal_size", **kwargs):
        super().__init__(name="pipeline_parallel", group="parallel", is_tunable=True, is_distributed=True)
        DistributedGraphMixin.__init__(self)
        self._shard_planner = shard_planner

    @property
    def shard_planner(self):
        if isinstance(self._shard_planner, str) and self._shard_planner == "equal_size":
            self._shard_planner = BaseStagePlanner()
        elif not isinstance(self._shard_planner, callable):
            logger.warning(f"shard planner {self._shard_planner} not supported")
            self._shard_planner = BaseStagePlanner()

        return self._shard_planner

    def tune(self, model_context, config=None, strategy=None, apply_transform=False):
        logger.info(
            "Must call destroy_parallel_group at the end of the training script. "
            "Must hande over the lr scheduler and optimizer to auto_accelerate."
        )

        # setting up pipeline related configs
        config = config if config else dict()
        pipe_ranks = config.get("pipe_ranks", parallel_group_and_ranks("pipe")[1])
        nstages = config.get("nstages", len(pipe_ranks))
        chunks = config.get("chunks", len(pipe_ranks))
        use_c10d = config.get("use_c10d", False)
        if not use_c10d:
            logger.warning(
                "Non-c10d Pipeline parallelism is more stable with Environment Variable: "
                "CUDA_DEVICE_MAX_CONNECTIONS=1"
            )
        dynamic_shape = config.get("dynamic_shape", False)
        subset_topo = self.device_topo.get_physical_topology(pipe_ranks)
        pipe_schedule = config.get("pipe_schedule", "Interleaved1F1B")
        # This is a hack to specify split points.
        insert_before_modules = config.get("insert_before_modules", None)
        self._shard_planner = config.get("shard_planner", self._shard_planner)

        # First rewrite the module
        train_mode = model_context.convert_to_loss_wrapper()
        output_loss_value_spec = True if train_mode else None
        model = model_context.model

        graph, sharding_specs, tensor_shapes = self._trace_and_propagate(model_context, config, strategy)

        checkpoint_keys = dict()

        for name, mod in model.named_modules():
            for param_name, param in mod.named_parameters():
                checkpoint_keys[f"{name}.{param_name}"] = getattr(param, "checkpoint_name", None)
            for buffer_name, mod_buffer in mod.named_buffers():
                checkpoint_keys[f"{name}.{buffer_name}"] = getattr(mod_buffer, "checkpoint_name", None)

        if insert_before_modules is not None:
            # match qualname with node name
            insert_before_nodes = []
            for node in graph.nodes:
                if node.op == "call_module" and node.target in insert_before_modules:
                    insert_before_nodes.append(node.name)
        else:
            insert_before_nodes = self.shard_planner.generate_sharding_plan(
                model, graph, sharding_specs, tensor_shapes, device_topo=subset_topo, nstages=nstages
            )
        # other compiler related args, ignore checkpoint
        # whether to checkpoint will be determined externally

        # The correct input_batch_size / example inputs can only be obtained at apply_wrapper
        # as in MixedParallel mode, no parallel group is created
        data_configs = {}
        # should not set down env configs as it is device specific
        env_configs = {}
        compiler_configs = {
            "insert_before_nodes": insert_before_nodes,
            "multi_use_param_spec": MultiUseParameterConfig.REPLICATE,
            "output_loss_value_spec": output_loss_value_spec,
            "expected_num_stages": nstages,
            "compile_with_dynamo": False,
            "export_mode": True,
            "chunks": chunks,
            "pipe_schedule": pipe_schedule,
            "checkpoint_keys": checkpoint_keys,
            "use_c10d": use_c10d,
            "train_mode": train_mode,
            "dynamic_shape": dynamic_shape,
        }

        best_config = {"data_configs": data_configs, "env_configs": env_configs, "compiler_configs": compiler_configs}
        if apply_transform:
            model_context = self.transform(model_context, config)
        return True, best_config, model_context

    def transform(self, model_context, config=None):
        config = {} if not config else config
        model_context.add_wrapper("pipe", PipelineParallelOptimization.apply_wrapper, config, is_pre_wrapper=True)
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config):
        try:
            # get environment settings
            _, pipe_ranks = parallel_group_and_ranks("pipe")
            _, data_ranks = parallel_group_and_ranks("data")
            device = f"cuda:{local_rank()}"

            # If the model is already a tp compiled model, quite tracing
            compiler_configs = wrapper_config.get("compiler_configs", dict())
            compiler_configs.setdefault("model_input_format", model_context.model_input_format)
            expected_num_stages = compiler_configs["expected_num_stages"]
            insert_before_nodes = compiler_configs["insert_before_nodes"]
            amp_config = compiler_configs.get("amp_config", None)

            # amp for fp16 is not compatible with InterleavedSchedule:
            if (
                len(pipe_ranks) < expected_num_stages
                and amp_config is not None
                and amp_config["dtype"] != torch.bfloat16
            ):
                logger.info("AMP Interleaved Schedule is only compatible with torch.bfloat16")
                amp_config["dtype"] = torch.bfloat16

            leaf_modules = compiler_configs.get("leaf_modules", list(_SHARDABLE_OPERATORS.values()))

            if not model_context.tp_status or not isinstance(model_context.model, torch.fx.GraphModule):
                # First rewrite the module
                model_context.convert_to_loss_wrapper(amp_config=amp_config)
                graph = model_context.capture_compute_graph(backend="meta_fx", leaf_modules=leaf_modules)
                pipe_graph = insert_split_point(graph, insert_before_nodes, expected_num_stages)
                gm = GraphModule(model_context.model, pipe_graph)
            else:
                gm = model_context.model
            # insert pipe split
            # In case the model is already tp compiled, we assume pipe splits already inserted
            if not model_context.tp_status:
                if not _check_pipe_split_inserted(gm):
                    pipe_graph = insert_split_point(gm.graph, insert_before_nodes, expected_num_stages)
                    gm = torch.fx.GraphModule(gm, pipe_graph)
                # Since a fake_gm is allocated on meta, we will default to storing it.
                counter = AutoAccelerateContext.counter
                # If fake_gm already exists, the pipe gm might already been modified by TP, do not use it.
                if not hasattr(AutoAccelerateContext, "fake_gm") or counter not in AutoAccelerateContext.fake_gm:
                    fake_gm = copy.deepcopy(gm).to("meta")

                    if not hasattr(AutoAccelerateContext, "fake_gm"):
                        AutoAccelerateContext.add_ac_attr("fake_gm", {counter: fake_gm})
                    else:
                        AutoAccelerateContext.fake_gm[counter] = fake_gm

            module_replace = compiler_configs.get("module_replace", False)

            # Must do module replace after tracing, because some replaced ops are not tracable
            # FIXME potentially it causes a mismatch between the graph and the model, disabled until a fix
            # FIXME Since we are operating on graph module, this may not be a problem. But tracing reduces the
            # available choices for module replace, how to fix this?
            if module_replace:
                _replace_by_config(gm, _get_default_replace_config(), False)
                _enable_flash_attn_by_attr(gm)

            data_configs = wrapper_config.get("data_configs", dict())
            # Data related configs has to be retrieved at this stage,
            # to make sure all parallel_groups are correctly created.
            data_configs["example_inputs"] = model_context.get_one_input_batch()
            # This should be the original input batch size to pipeline model,
            # as we are using this to judge the correct batch dimension.
            data_configs["input_batch_size"] = model_context.get_input_batch_size()

            env_configs = wrapper_config.get("env_configs", dict())
            env_configs.setdefault("pipe_ranks", pipe_ranks)
            env_configs.setdefault("data_ranks", data_ranks)
            env_configs.setdefault("device", device)

            pipe_model = pippy_compiler(gm, data_configs, env_configs, compiler_configs)
            model_context.model = pipe_model
            if model_context.loss_func is not None:
                model_context.loss_func = pipe_loss_func
            logger.info(f"Pipe rank {rank()} wait for other ranks for initialization")
            torch.distributed.rpc.api._wait_all_workers()
            logger.info(f"Pipe rank {rank()} Successfully transform the model into pipeline parallel model.")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Failed to transform the model into a parallel model, with error: {e}")
