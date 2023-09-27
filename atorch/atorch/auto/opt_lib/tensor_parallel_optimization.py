import copy
import gc
import os
import shutil
import traceback

import torch

from atorch import local_rank
from atorch.auto.opt_lib.optimization import DistributedGraphMixin, Optimization
from atorch.auto.opt_lib.shard_planners import BaseTensorParallelPlanner, MIPTensorParallelPlanner
from atorch.auto.opt_lib.utils import find_memory_factor, print_sharding_specs
from atorch.common.log_utils import default_logger as logger
from atorch.distributed.distributed import parallel_group_and_ranks
from atorch.modules.distributed_modules.compilers import tp_compiler
from atorch.modules.distributed_modules.materialize_modules import materialize_modules_to_device
from atorch.modules.distributed_modules.modules_registry import _SHARDABLE_OPERATORS, _register_custom_operators
from atorch.utils.meta_model_utils import _MetaModeContext, empty_param, is_meta, recursive_empty_param


# FIXME Support manual leaf modules, supports manual strategy
# FIXME Implement more strategies, support manual selection of strategies
class TensorParallelOptimization(Optimization, DistributedGraphMixin):
    def __init__(self, shard_planner="mip", **kwargs):
        """
        Args:
            shard_planner: the shard planner to be used, can be string, in which case
                'base'/None -> BaseTensorParallelPlanner
                'mip' -> MIPTensorParallelPlanner
        """
        super().__init__(name="tensor_parallel", group="parallel", is_tunable=True, is_distributed=True)
        DistributedGraphMixin.__init__(self)
        self._shard_planner = shard_planner

    @property
    def shard_planner(self):
        if self._shard_planner is None or self._shard_planner == "base":
            self._shard_planner = BaseTensorParallelPlanner()
        if self._shard_planner == "mip":
            self._shard_planner = MIPTensorParallelPlanner(
                memory_bound=self.memory_bound,
                fp32_flops=self.fp32_flops,
                merge_nodes=True,
                solver="glpk",
                greedy_init=True,
                timelimit=10 * 60,
            )
        return self._shard_planner

    # In case of mixed parallel mode, set a pipe graph
    def tune(
        self,
        model_context,
        config=None,
        strategy=None,
        apply_transform=False,
        time_limit=None,
    ):
        try:
            _register_custom_operators()

            config = config if config is not None else dict()
            if "replacement_map" in config:
                replacement_map = config["replacement_map"]
                logger.info(f"sharding plan already specified: {replacement_map}, skip tunning")
                return True, config, model_context

            self._shard_planner = config.get("shard_planner", self._shard_planner)

            tp_debug = logger.root.level > 30

            tp_ranks = config.get("tp_ranks", parallel_group_and_ranks("tensor")[1])
            subset_topo = self.device_topo.get_physical_topology(tp_ranks)

            graph, sharding_specs, tensor_shapes = self._trace_and_propagate(model_context, config, strategy)

            optimizer = model_context.create_optim()
            model = model_context.model

            if isinstance(self.shard_planner, MIPTensorParallelPlanner):
                # adjust memory factor for mip planner
                # FIXME not checking checkpointing_optimization now, since not sure how it affects GPURAM
                forward_factor, grad_factor, param_factor, optimizer_factor = find_memory_factor(strategy)
                self.shard_planner.set_mem_factors(forward_factor, grad_factor, param_factor, optimizer_factor)

            best_config = self.shard_planner.generate_sharding_plan(
                model, graph, sharding_specs, tensor_shapes, subset_topo, optimizer
            )
            if tp_debug:
                replaced_specs = best_config["replaced_specs"]
                logger.info("[TP DEBUG]: Print Graph Replaced Specs")
                print_sharding_specs(replaced_specs)

            del optimizer
            gc.collect()
            logger.info(f"{local_rank()} finish tuning TP Models")
            return True, best_config, model_context
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Failed to generate a sharding plan, with error: {e}")
            best_config = dict()
            return False, best_config, model_context

    def transform(self, model_context, config=None):
        model_context.add_wrapper("tp", TensorParallelOptimization.apply_wrapper, config, is_pre_wrapper=True)
        return model_context

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config):
        """
        args:
            wrapper_config:
                replacement_map: the map from node name to the distributed implementation's name,
                    each distributed implementation must specifies an input_spec and an output_spec
                process_groups: process groups to be initialized
                replaced_specs: input and output specs for nodes after replacement.
                    see atorch.auto.opt_lib.auto_tensor_parallel_strategies.base_strategy for details
                changed_local_nodes: local nodes that specs changed according to input nodes
                process_group_assignment: a dict describing where to place every parallel operators
                gm: the graph module to be used if pipeline is enabled
                leaf_modules: this controls the leaf modules for tracer
                defer_init: whether to defer tp initialization, used when TP is mixed with FSDP/Pipeline
        return:
            A tensor parallel model
        """
        try:
            compiler_configs = wrapper_config
            gm = compiler_configs.get("gm", None)
            defer_init = compiler_configs.get("defer_init", False)
            leaf_modules = compiler_configs.get("leaf_modules", list(_SHARDABLE_OPERATORS.values()))
            # In case of mixed parallel mode, gm is created and preprocessed by MP.
            gm = gm if gm else model_context.export_graph_module(backend="meta_fx", leaf_modules=leaf_modules)
            if defer_init and not is_meta(gm):
                # If we have to defer init, we must move everything onto META, this is required by FSDP
                if (
                    torch.distributed.is_initialized() and int(local_rank()) == 0
                ) or not torch.distributed.is_initialized():
                    if not os.path.isdir(_MetaModeContext.offload_path):
                        os.makedirs(_MetaModeContext.offload_path)
                    else:
                        try:
                            shutil.rmtree(_MetaModeContext.offload_path)
                            os.makedirs(_MetaModeContext.offload_path)
                        except OSError as e:
                            logger.warning(f"failed to remove offload directory, with error: {e}")
                empty_param(gm)
                recursive_empty_param(gm)
                torch.distributed.barrier()
            tp_model = tp_compiler(gm, copy.deepcopy(compiler_configs))

            # If pipeline parallelism also enabled, wait until pipe stage init
            # to move stage model into GPU.
            if not defer_init:
                device = torch.device(type="cuda", index=local_rank())
                materialize_modules_to_device(tp_model, device)
                tp_model.to(device)

            model_context.model = tp_model
            model_context.update_tp_status(True)
            logger.info("Successfully transformed the model into tensor parallel model")
        except Exception as e:
            traceback.print_exc()
            logger.warning(f"Failed to transform the model into a parallel model, with error: {e}")

    @staticmethod
    def reset(config):
        # this should remove the meta model offload dir at correct timing
        pass
