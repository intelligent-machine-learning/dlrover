try:
    from pippy.IR import pipe_split
except ImportError:
    pipe_split = None

from atorch.modules.distributed_modules.modules_registry import _SHARD_MAP, _SHARDABLE_OPERATORS, _SHARDED_OPERATORS
from atorch.utils.graph_transform_utils import map_aggregate


# FIXME This is a temporary fix for FSDP and TP compatibility,
# TP will only trace the module to the level of FSDP wrapping, with constraint of being shardable.
# The wrapping will be broken if the wrapped op is shardable, causing FSDP to be slow (in most cases).
def _propose_leaf_modules(atorch_wrap_cls=None):
    leaf_modules = None
    if atorch_wrap_cls is not None:
        leaf_modules = list(set(_SHARDABLE_OPERATORS.values()) & set(atorch_wrap_cls))
    if leaf_modules is None or len(leaf_modules) == 0:
        leaf_modules = list(_SHARDABLE_OPERATORS.values())
    return leaf_modules


def _propose_wrap_cls(leaf_modules=set()):
    leaf_module_names = [name for name, cls in _SHARDABLE_OPERATORS.items() if cls in leaf_modules]
    if len(leaf_modules) != 0:
        atorch_wrap_cls = {
            op
            for name in leaf_module_names
            for op in [_SHARDABLE_OPERATORS[name]] + [_SHARDED_OPERATORS[shard] for shard in _SHARD_MAP[name]]
        }
    else:
        atorch_wrap_cls = None
    if len(leaf_modules) == len(list(_SHARDABLE_OPERATORS.items())):
        # in this case, auto wrap is meaning less
        atorch_wrap_cls = None
    return atorch_wrap_cls


def propose_leaf_modules_by_strategy(model, strategy=None):
    wrap_cls = None
    if strategy is None:
        return _propose_leaf_modules(wrap_cls)
    for opt in strategy:
        opt_name = opt[0]
        if opt_name == "fsdp":
            if len(opt) > 1:
                opt_config = opt[1]
                atorch_wrap_cls = set(to_module_class_by_name(model, opt_config.get("atorch_wrap_cls", set())))
                if wrap_cls is None:
                    wrap_cls = atorch_wrap_cls
                else:
                    wrap_cls = wrap_cls & atorch_wrap_cls
        if opt_name == "checkpoint":
            if len(opt) > 1:
                opt_config = opt[1]
                ckpt_wrap_cls = set(to_module_class_by_name(model, opt_config))
                if wrap_cls is None:
                    wrap_cls = ckpt_wrap_cls
                else:
                    wrap_cls = wrap_cls & ckpt_wrap_cls

    leaf_modules = _propose_leaf_modules(wrap_cls)
    return leaf_modules


def find_modules(model, m_list):
    if isinstance(model, tuple(m_list)):
        return [model]
    res = []
    for mo in model.modules():
        if isinstance(mo, tuple(m_list)):
            res.append(mo)
    return res


# Convert module name in module_list into module type.
# Also ignore any module names that not exist in model.
def to_module_class_by_name(model, module_list):
    module_classes = {}
    for m in module_list:
        if type(m) is str:
            module_classes[m] = None
    unassigned_num = len(module_classes)
    if unassigned_num > 0:
        for m in model.modules():
            if type(m).__name__ in module_classes.keys() and module_classes[type(m).__name__] is None:
                module_classes[type(m).__name__] = type(m)
                unassigned_num -= 1
                if unassigned_num == 0:
                    break
    result = []
    for m in module_list:
        if type(m) is str:
            if module_classes[m] is not None:
                result.append(module_classes[m])
        else:
            result.append(m)
    return type(module_list)(result)


def insert_split_point(graph, insert_before_nodes, expected_num_stages):
    nstages = 1
    insert_before_nodes = [node for node in graph.nodes if node.name in insert_before_nodes]
    for node in insert_before_nodes:
        if nstages == expected_num_stages:
            break
        with graph.inserting_before(node):
            graph.call_function(pipe_split, (), {})
        nstages += 1

    return graph


def find_memory_factor(strategy):
    if strategy is None:
        return 1.0, 1.0, 1.0, 1.0
    _, parallel_mode_config = strategy.get_parallel_mode()
    zero_size = None
    data_size = None
    if parallel_mode_config:
        for group_name, group_size in parallel_mode_config[0]:
            if group_name == "data":
                data_size = group_size
            if group_name == "zero":
                zero_size = group_size
    if zero_size is None and data_size is not None:
        zero_size = data_size
    else:
        zero_size = 1
    zero_size = zero_size if zero_size else 1
    data_size = data_size if data_size else 1
    # TP musr have parallel_mode_config
    forward_factor = 1.0
    grad_factor = 1.0
    param_factor = 1.0
    optimizer_factor = 1.0
    opt_names = list(opt[0] for opt in strategy)
    for opt_name in opt_names:
        if opt_name == "amp_native":
            forward_factor *= 0.5
            if "fsdp" in opt_names or "zero2" in opt_names:
                grad_factor *= 0.5
            param_factor *= 1.5
        if opt_name == "zero1":
            optimizer_factor *= 1 / zero_size
        if opt_name == "zero2":
            optimizer_factor *= 1 / zero_size
            grad_factor *= 1 / zero_size
        if opt_name == "fsdp":
            optimizer_factor *= 1 / zero_size
            grad_factor *= 1 / zero_size
            param_factor *= 1 / zero_size

    return forward_factor, grad_factor, param_factor, optimizer_factor


def print_sharding_specs(sharding_specs):
    """Given the sharding specs of a graph or a subgraph, print the sharding spec of each node"""

    def spec_to_str(sharding_spec):
        return str(sharding_spec)

    for node, spec_dict in sharding_specs.items():
        print(f"node: {node}")
        input_spec_dict = spec_dict["input_spec"]
        output_spec_str = map_aggregate(spec_dict["output_spec"], spec_to_str)

        print("----input")
        for input_node, input_spec in input_spec_dict.items():
            spec_str = map_aggregate(input_spec, spec_to_str)
            print(f"        {input_node}: {spec_str}")
        print("----output")
        print(f"        {output_spec_str}")
