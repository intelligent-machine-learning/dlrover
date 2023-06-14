# A Tensor Parallel Compiler Relying on replacement map etc.
# The distributed runtime relies on raw torch.Tensor.
# This compiler replaces all nodes specified in replacement map
# TODO The distributed IR should be more consistent with a new NodeHandler type strategy generator
import traceback
from functools import wraps
from typing import Any, Dict, Set, Tuple

from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

try:
    from torch.fx.passes.utils.fuser_utils import topo_sort
except ImportError:
    from atorch.utils.graph_transform_utils import topo_sort

from atorch.common.log_utils import default_logger as logger
from atorch.modules.distributed_modules.mappings_registry import get_communicator
from atorch.modules.distributed_modules.modules_registry import _SHARDED_OPERATORS
from atorch.utils.graph_transform_utils import _replace_input_node, _replace_node_module
from atorch.utils.sharding_spec import shard_eq


def _insert_communicator(
    graph,
    prev_shard,
    cur_shard,
    appended_communicators,
    input_node,
    user_node,
    sharding_specs,
):
    # FIXME need a safer way to judge the need of resharding
    if prev_shard is not None and cur_shard is not None and not shard_eq(prev_shard, cur_shard):
        communicator = get_communicator(prev_shard, cur_shard)
        if (prev_shard, cur_shard) in appended_communicators.setdefault(input_node.name, dict()).keys():
            new_input_node = appended_communicators[input_node.name][(prev_shard, cur_shard)]
            _replace_input_node(input_node, new_input_node, user_node)
        else:
            # insert after input node so the pipe split nodes will not break a submodel
            with graph.inserting_after(input_node):
                new_input_node = graph.call_function(communicator)
                sharding_specs[new_input_node.name] = dict()
                sharding_specs[new_input_node.name]["input_spec"] = {input_node.name: prev_shard}
                sharding_specs[new_input_node.name]["output_spec"] = cur_shard
                _replace_input_node(input_node, new_input_node, user_node)
                # rewrite user_node's input_spec
                sharding_specs[user_node.name]["input_spec"].pop(input_node.name, None)
                sharding_specs[user_node.name]["input_spec"][new_input_node.name] = cur_shard
                appended_communicators[input_node.name][(prev_shard, cur_shard)] = new_input_node


# Support partial compilation
# Will partial compilation cause problem for replicate parameters?
def tp_compiler(gm, compiler_configs):
    """
    args:
        gm (torch.fx.GraphModule): a graph module to be compiled
        compiler_configs:
            replacement_map: the map from node name to the distributed implementation's name,
                each distributed implementation must specifies an input_spec and an output_spec
            process_groups: process groups to be initialized
            replaced_specs: input and output specs for nodes after replacement.
                see atorch.auto.opt_lib.auto_tensor_parallel_strategies.base_strategy for details
            changed_local_nodes: local nodes that specs changed according to input nodes
            process_group_assignment: a dict describing where to place every parallel operators
            defer_init: whether to defer tp initialization, to be used if mixed with FSDP/Pipeline

    return:
        A tensor parallel model
    """
    try:
        model = gm
        graph = gm.graph

        modules = dict(model.named_modules())
        appended_communicators: Dict[str, Dict[Tuple, Node]] = dict()

        replacement_map: Dict[str, str] = compiler_configs.get("replacement_map", dict())
        changed_local_nodes: Set[str] = compiler_configs.get("changed_local_nodes", set())
        replaced_specs: Dict[str, Any] = compiler_configs.get("replaced_specs", dict())
        process_group_assignment: Dict[str, Any] = compiler_configs.get("process_group_assignment", dict())
        topo_nodes = topo_sort(graph.nodes)
        defer_init = compiler_configs.get("defer_init", False)

        for node in topo_nodes:
            if node.name in replacement_map.keys():
                if node.op != "call_function" and node.op != "call_module":
                    continue

                if node.op == "call_function":
                    orig_target = node.target
                else:
                    orig_target = model.get_submodule(node.target)

                replacement_target_name = replacement_map.get(node.name, None)
                if replacement_target_name is not None:
                    replacement_target = _SHARDED_OPERATORS[replacement_target_name]

                node_group = process_group_assignment[node.name].get("group", None)
                node_ranks = process_group_assignment[node.name].get("ranks", None)
                if node.op == "call_function":

                    def wrap_replacement_target(func, process_group="tensor", ranks=None):
                        @wraps(func)
                        def group_replacement_target(*args, **kwargs):
                            return replacement_target(*args, **kwargs, process_group=process_group, ranks=ranks)

                        return group_replacement_target

                    node.target = wrap_replacement_target(orig_target, node_group, node_ranks)
                else:
                    replacement_module = replacement_target(
                        orig_module=orig_target, process_group=node_group, ranks=node_ranks, defer_init=defer_init
                    )
                    _replace_node_module(node, modules, replacement_module)

        # Tracking nodes in topological order helps to make sure we insert communicators
        # in the correct order
        topo_nodes = topo_sort(graph.nodes)
        for node in topo_nodes:
            if node.name in replacement_map.keys() or node.name in changed_local_nodes:
                # insert communicator for input nodes
                for input_node in list(node.all_input_nodes):
                    prev_shard = replaced_specs.get(input_node.name, dict()).get("output_spec", None)
                    cur_shard = replaced_specs.get(node.name, dict()).get("input_spec", None).get(input_node.name, None)
                    _insert_communicator(
                        graph,
                        prev_shard,
                        cur_shard,
                        appended_communicators,
                        input_node,
                        node,
                        replaced_specs,
                    )

                # insert comminicator for users
                for user_node in list(node.users):
                    if not (user_node.op == "call_function" and user_node.target.__name__ == "getitem"):
                        prev_shard = replaced_specs.get(node.name, dict()).get("output_spec", None)
                        cur_shard = (
                            replaced_specs.get(user_node.name, dict()).get("input_spec", None).get(node.name, None)
                        )
                        _insert_communicator(
                            graph,
                            prev_shard,
                            cur_shard,
                            appended_communicators,
                            node,
                            user_node,
                            replaced_specs,
                        )

        tp_model = GraphModule(model, graph)
        return tp_model

    except Exception as e:
        traceback.print_exc()
        logger.warning(f"Failed to transform the model into a parallel model, with error: {e}")
