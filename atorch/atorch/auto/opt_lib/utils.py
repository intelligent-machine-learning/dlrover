try:
    from pippy.IR import pipe_split
except ImportError:
    pipe_split = None

from atorch.utils.graph_transform_utils import map_aggregate


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
        if type(m) == str:
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
        if type(m) == str:
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
        if opt_name == "amp_native" or opt_name == "amp_apex_o2":
            forward_factor *= 0.5
            if "fsdp" in opt_names or "zero2" in opt_names:
                grad_factor *= 0.5
            param_factor *= 1.5
        if opt_name == "amp_apex_o1":
            forward_factor *= 0.5
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
