import inspect
from abc import ABC, abstractmethod

import torch
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import ShapeProp

from atorch import local_rank, world_size
from atorch.auto.device_context import get_device_context
from atorch.auto.opt_lib.shard_planners import SimpleTopology
from atorch.auto.opt_lib.utils import print_sharding_specs, propose_leaf_modules_by_strategy, to_module_class_by_name
from atorch.common.log_utils import default_logger as logger
from atorch.utils.graph_transform_utils import map_aggregate
from atorch.utils.meta_model_utils import reload_meta_module
from atorch.utils.shape_prop import MetaShapeProp
from atorch.utils.spec_prop import MetaSpecProp, SpecProp
from atorch.utils.version import torch_version

try:
    from torch._subclasses.fake_tensor import FakeTensorMode
except ImportError:
    FakeTensorMode = object


class Optimization(ABC):
    def __init__(self, name, group, is_tunable, is_distributed=False):
        self.name = name
        self.group = group
        self.is_tunable = is_tunable
        self.is_distributed = is_distributed

    def distributed_only(self, config=None):
        return self.is_distributed

    @abstractmethod
    def tune(self, model_context, config=None, strategy=None, apply_transform=True, time_limit=None):
        """
        Find best config for this optimization for model_context.
        config: an imcomplete config or None.
        strategy: a list of optimization methods, and self.name is in it.
                  all methods before self.name are already applied on model_context
                  all methods after self.name will be applied after
        apply_transform: if apply transform to model_context with apply_wrapper=False
        time_limit: if not None, should finish tune within this seconds
        Return: status, best_config, model_context
            status: if tune successfully.
            best_config: the successfully tuned config for this optimization
            model_context: if apply_transform, transformed model_context; original model_context otherwise.
        """
        pass

    @abstractmethod
    def transform(self, model_context, config=None):
        """
        Return the transformed model_context.
        Note that is some transformation must applied after all other optimization methods
        are done, or the transformation must use instanced model/optim/dataloader, a wrapper
        can be added to model_context, so the transformation will be applied later.
        """
        pass

    @staticmethod
    def apply_wrapper(model_context, wrapper_name, wrapper_config=None):
        """
        Apply a wrapper (whose name is wrapper_name) with wrapper_config to model_context
        and return the wrappered model_context
        """
        return model_context

    @staticmethod
    def reset(config):
        """Reset environment changed by current optimization."""
        pass

    @staticmethod
    def device_supported(config=None, device_capability=None):
        "If the method with config is supported by device with device_capability"
        return True


class DistributedGraphMixin:
    def __init__(
        self,
        num_nodes=None,
        num_devices_per_node=None,
        tracer_backend="meta_fx",
        prop_mode="interpreter",
        use_fake_mode=False,
        device_context=None,
    ):
        """
        This is a mixin encapsulating the functionality of device topology abstraction, graph extraction,
        graph interpretation etc. This mixin is primarily used in places where shape propogation is needed.

        Args:
            num_nodes: number of nodes in the cluster.
            num_devices_per_node: self explained.
            tracer_backend: the backend to be used for tracing, support fx, meta_fx, and dynamo.
            prop_mode: mode for shape inference, by default interpreter. Supports also a faster version: "meta_tracer"
            use_fake_mode: Whether to use fake mode for interpreter
        """
        if num_nodes is None or num_devices_per_node is None:
            if torch.distributed.is_initialized():
                num_devices_per_node = torch.cuda.device_count()
                if num_devices_per_node is not None and num_devices_per_node != 0:
                    num_nodes = max(world_size() // num_devices_per_node, 1)
                else:
                    num_nodes = 1
            else:
                num_devices_per_node = 1
                num_nodes = 1
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        device_context = get_device_context() if device_context is None else device_context
        intra_node_bandwidth = device_context.intra_node_bandwidth
        inter_node_bandwidth = device_context.inter_node_bandwidth
        self.fp32_flops = device_context.fp32_flops
        self.memory_bound = device_context.gpu_memory
        self.device_topo = SimpleTopology(
            num_nodes,
            num_devices_per_node,
            intra_node_bandwidth=intra_node_bandwidth,
            inter_node_bandwidth=inter_node_bandwidth,
        )
        self.prop_mode = prop_mode
        self.tracer_backend = tracer_backend
        self.use_fake_mode = use_fake_mode

    def _apply_interpreter(self, model, graph, input_batch, InterpreterCls, use_fake_mode=False):
        available_device = (
            torch.device(type="cuda", index=local_rank()) if torch.cuda.is_available() else torch.device("cpu")
        )
        device = next(model.parameters()).device if use_fake_mode else available_device
        traced_model = GraphModule(model, graph)
        sig = inspect.signature(traced_model.forward)
        default_names = sig.parameters.keys() - input_batch.keys()
        default_args = {p.name: p.default for p in sig.parameters.values() if p.name in default_names}
        input_batch.update(default_args)

        fake_mode = FakeTensorMode(allow_non_fake_inputs=True) if torch_version() >= (2, 0, 0) else FakeTensorMode()

        def _lower_tensor_to_device(input_):
            lowered_input = input_.to(device) if isinstance(input_, torch.Tensor) else input_
            return fake_mode.from_tensor(lowered_input) if use_fake_mode else lowered_input

        input_batch = {k: map_aggregate(v, _lower_tensor_to_device) for k, v in input_batch.items()}

        input_data = list()
        for p in sig.parameters.values():
            input_data.append(input_batch[p.name])
        input_data = tuple(input_data)
        interpreter = (
            InterpreterCls(traced_model, fake_mode=fake_mode)
            if use_fake_mode
            else InterpreterCls(traced_model, device=device)
        )
        interpreter.propagate(*input_data)
        return traced_model

    def _get_tensors_shapes(self, model, graph, input_batch, apply_interpreter=True, use_fake_mode=False):
        if apply_interpreter:
            interpreter_cls = ShapeProp if use_fake_mode else MetaShapeProp
            traced_model = self._apply_interpreter(model, graph, input_batch, interpreter_cls, use_fake_mode)
        else:
            traced_model = GraphModule(model, graph) if not isinstance(model, GraphModule) else model

        tensor_shapes = dict()
        for node in traced_model.graph.nodes:
            tensor_shape = node.meta["tensor_meta"] if "tensor_meta" in node.meta else None

            tensor_shapes[node.name] = tensor_shape

        return tensor_shapes

    def _get_sharding_specs(self, model, graph, input_batch, apply_interpreter=True, use_fake_mode=False):
        if apply_interpreter:
            interpreter_cls = SpecProp if use_fake_mode else MetaSpecProp
            traced_model = self._apply_interpreter(model, graph, input_batch, interpreter_cls, use_fake_mode)
        else:
            traced_model = GraphModule(model, graph) if not isinstance(model, GraphModule) else model

        sharding_specs = dict()
        for node in traced_model.graph.nodes:
            input_spec = node.meta["input_sharding_spec"]
            output_spec = node.meta["output_sharding_spec"]

            sharding_specs[node.name] = dict()
            sharding_specs[node.name]["input_spec"] = input_spec
            sharding_specs[node.name]["output_spec"] = output_spec
        return sharding_specs

    def _trace_and_propagate(
        self,
        model_context,
        config=None,
        strategy=None,
        parallel_config=None,
    ):

        config = config if config is not None else dict()
        self.prop_mode = config.get("prop_mode", self.prop_mode)
        self.use_fake_mode = config.get("use_fake_mode", self.use_fake_mode)
        self.tracer_backend = config.get("tracer_backend", self.tracer_backend)
        if self.prop_mode == "meta_tracer":
            self.tracer_backend = "meta_fx"
            logger.warning(
                "Using meta_tracer mode for shape inference and spec prop, "
                "this may not work properly if inplace operators such as torch.nn.functional.relu is used"
            )

        tp_debug = logger.root.level > 30

        model = model_context.model
        input_batch = model_context.get_one_input_batch(
            need_model_input_correspondence=True, parallel_config=parallel_config
        )
        # TP could destroy module structure based on which FSDP/checkpointing works
        # If graph is givem, TP will not check the consistence between the model and the graph

        if "leaf_modules" not in config:
            leaf_modules = propose_leaf_modules_by_strategy(model, strategy)
        else:
            logger.info(
                "Setting leaf modules manually could cause problem in torch2.0, "
                "if this happens, try setting a mock implementation for the leaf modules with: "
                "register_meta_overrides(orig_target, meta_target)"
            )
            leaf_modules = config["leaf_modules"]
            leaf_modules = to_module_class_by_name(model, leaf_modules)

        graph = model_context.capture_compute_graph(
            backend=self.tracer_backend, leaf_modules=leaf_modules, parallel_config=parallel_config
        )

        apply_interpreter = self.prop_mode == "interpreter" or self.tracer_backend != "meta_fx"
        if apply_interpreter:
            reload_meta_module(model)
        sharding_specs = self._get_sharding_specs(model, graph, input_batch, apply_interpreter, self.use_fake_mode)

        if tp_debug:
            logger.info("[DGOPT DEBUG]: Print Graph")
            graph.print_tabular()
            logger.info("[DGOPT DEBUG]: Print Graph Sharding Specs")
            print_sharding_specs(sharding_specs)

        tensor_shapes = self._get_tensors_shapes(model, graph, input_batch, apply_interpreter, self.use_fake_mode)
        return graph, sharding_specs, tensor_shapes
