# This implements a size based sharding strategy
# Mainly adopted PiPPy implementation

import torch

try:
    from pippy.IR import pipe_split
except ImportError:
    pipe_split = None

from atorch.common.log_utils import default_logger as logger


def _analyze_node_size(model, graph):
    # state_dict helps us to get parameter sizes
    state_dict = model.state_dict()

    # Function Parameter Usage
    node_param_sizes = {}
    for node in graph.nodes:
        if node.op == "get_attr":  # a parameter node
            param_name = node.target
            assert param_name in state_dict
            param = state_dict[param_name]
            # Find use site of this parameter
            for user in node.users:
                func_param_sizes = node_param_sizes.setdefault(user, {})
                func_param_sizes.setdefault(param_name, param.numel())

    # Module Parameter Usage
    for node in graph.nodes:
        # We calcuate size of a user-defined submodule as a whole
        if node.op == "call_module":
            mod_param_sizes = {}
            submod: torch.nn.Module = model.get_submodule(node.target)
            for param_name, param in submod.named_parameters():
                mod_param_sizes.setdefault(param_name, param.numel())
            if mod_param_sizes:
                node_param_sizes.setdefault(node, mod_param_sizes)

    for node, param_sizes in node_param_sizes.items():
        logger.debug(f"{node} has params: {param_sizes}")

    return node_param_sizes


def _get_total_size(gm):
    param_size = 0
    for param in gm.parameters():
        param_size += param.numel()
    buffer_size = 0
    for buffer in gm.buffers():
        buffer_size += buffer.numel()

    total_size = param_size + buffer_size
    return total_size


# Compute the pipe split insertion point without actually inserting the split,
# so as to keep the generated config serializable (fx.Graph is not).
# Note that the number of split stages is not guaranteed.
def _split_on_size_threshold(model, graph, threshold):
    # Analyze size of parameters/buffers used by each node in the graph
    node_param_sizes = _analyze_node_size(model, graph)

    # Record split positions
    insert_before_nodes = []

    # we can only store node.name because the node is not serializable
    def new_stage_before(node):
        insert_before_nodes.append(node.name)

    # Track the parameters we have seen in the current bucket and their total size
    accumulate_size = 0
    accumulate_params = {}

    for node in graph.nodes:
        if node not in node_param_sizes:
            # The callsite of this node does not involve parameters or buffers
            continue

        # Track the new parameters we see at this node as well as parameters that we have seen in current bucket
        new_size = 0
        new_params = {}
        repeated_size = 0
        repeated_params = {}
        param_sizes = node_param_sizes[node]
        if node.op == "call_function":
            # For function, the parameter it uses can be shared with other functions seen previously
            for param_name, size in param_sizes.items():
                if param_name not in accumulate_params:  # new parameter
                    new_params.setdefault(param_name)
                    new_size += size
                else:  # repeated parameter; mark down; use later
                    repeated_params.setdefault(param_name)
                    repeated_size += size
        elif node.op == "call_module":
            # For module, we count its paramters as a single whole
            for param_name, size in param_sizes.items():
                new_size += size

        if accumulate_size + new_size <= threshold:  # can accommodate this node in current bucket
            accumulate_size += new_size
            accumulate_params.update(new_params)
        elif accumulate_size == 0 and new_size > threshold:  # this node becomes a stage
            new_stage_before(node.next)
        else:  # cannot accommodate this node
            new_stage_before(node)
            accumulate_size = repeated_size + new_size
            accumulate_params.clear()
            accumulate_params.update(repeated_params)
            accumulate_params.update(new_params)

    # Insert pipe_split nodes at the recorded positions
    return insert_before_nodes


def split_into_nstages_equal_size(model, graph, nstages):
    total_size = _get_total_size(model)
    per_stage_size = total_size // nstages
    logger.debug(f"Total model size: {total_size}, " f"per stage size: {per_stage_size}")
    return _split_on_size_threshold(model, graph, per_stage_size)


class BaseStagePlanner(object):
    def __init__(self, strategy_name="base"):
        self.strategy_name = strategy_name

    def generate_sharding_plan(
        self,
        model,
        graph,
        sharding_specs=None,
        tensor_shapes=None,
        device_topo=None,
        optimizer=None,
        nstages=None,
        **kwargs,
    ):
        nstages = device_topo.num_devices() if nstages is None else nstages
        return split_into_nstages_equal_size(model, graph, nstages)
