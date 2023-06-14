from queue import SimpleQueue

from torch.fx.node import map_arg


def _parent_name(target):
    """Get the qualified name of the parent module of submodule 'target'

    Args:
        target (str): target attribute of a call_module node (which is a str)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def _replace_node_module(node, modules, replacement_module):
    """Replace the node's original target module by replacement_module.

    Args:
        node (torch.fx.node.Node): a graph call_module node whose target
            is to be replaced by replacement_module.
        modules (list[torch.nn.Modules]): a list of modules, all named submodules
            of the root moduel.
        replacement_module (torch.nn.Module): the module to replace the node's
            original target.
    """
    parent_name, name = _parent_name(node.target)
    modules[node.target] = replacement_module
    setattr(modules[parent_name], name, replacement_module)


def _replace_input_node(input_node, new_input_node, user_node):
    """Replace the user_node's input_node by new_input_node.

    Args:
        input_node: the input_node to be replaced.
        new_input_node: the new input node.
        user_node: the user node whose input_node is to be
            replaced by new_input_node
    """

    def maybe_replace_node(n):
        if n == input_node:
            return new_input_node
        else:
            return n

    new_args = map_arg(user_node.args, maybe_replace_node)
    new_kwargs = map_arg(user_node.kwargs, maybe_replace_node)
    assert isinstance(new_args, tuple)
    assert isinstance(new_kwargs, dict)
    user_node.args = new_args
    user_node.kwargs = new_kwargs
    new_input_node.args = (input_node,)


def topo_sort(nodes):
    """Topologically sort the nodes. Backported from torch.fx.passes.utils.fuser_utils (1.13)

    Args:
        nodes: a list of graph node

    Return:
        sorted_nodes: a list of node, in topological order.
    """
    # sort nodes according to the topological order
    indegree_map = {node: 0 for node in nodes}
    candidates = SimpleQueue()

    for node in nodes:
        for n in node.all_input_nodes:
            if n in indegree_map:
                indegree_map[node] += 1
        if indegree_map[node] == 0:
            candidates.put(node)

    sorted_nodes = list()
    while not candidates.empty():
        node = candidates.get()
        sorted_nodes.append(node)

        for n in node.users:
            if n in indegree_map:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    candidates.put(n)

    assert len(nodes) == len(sorted_nodes), "topological sorted nodes doesn't have same length as input nodes"

    return sorted_nodes


def map_aggregate(input_, fn):
    """
    Apply fn to each element in input_. input_ may be a list, tuple, slice, or dict with string keys.

    Args:
        input_: the input whose elements are to be mapped by fn
        fn: the Callable to transform the elements

    Returns:
        The input_ whose elements are transformed by fn
    """
    if isinstance(input_, tuple):
        t = tuple(map_aggregate(elem, fn) for elem in input_)
        # Support NamedTuple (if it has `_fields`) by repacking into original type.
        return t if not hasattr(input_, "_fields") else type(input_)(*t)
    elif isinstance(input_, list):
        return list(map_aggregate(elem, fn) for elem in input_)
    elif isinstance(input_, dict):
        return dict((k, map_aggregate(v, fn)) for k, v in input_.items())
    elif isinstance(input_, slice):
        return slice(map_aggregate(input_.start, fn), map_aggregate(input_.stop, fn), map_aggregate(input_.step, fn))
    else:
        return fn(input_)


def combine_map_aggregate(input_0, input_1, fn):
    """
    Traverse input_0 and input_1 simultaneously and apply fn to each pair of elements.

    Args:
        input_0, input_1: the inputs to be traversed
        fn: the Callable to transform the elements

    Returns:
        The transformed input_0
    """
    if isinstance(input_0, tuple) and isinstance(input_1, tuple):
        return tuple(combine_map_aggregate(a, b, fn) for a, b in zip(input_0, input_1))
    elif isinstance(input_0, list) and isinstance(input_1, list):
        for i in range(len(input_0)):
            input_0[i] = combine_map_aggregate(input_0[i], input_1[i], fn)
    elif isinstance(input_0, dict) and isinstance(input_1, dict):
        for k in input_0:
            input_0[k] = combine_map_aggregate(input_0[k], input_1[k], fn)
    elif isinstance(input_0, slice) and isinstance(input_1, slice):
        return slice(
            combine_map_aggregate(input_0.start, input_1.start, fn),
            combine_map_aggregate(input_0.stop, input_1.stop, fn),
            combine_map_aggregate(input_0.step, input_1.step, fn),
        )
    else:
        fn(input_0, input_1)
        return input_0
    return input_0


# _pack and _unpack methods are backported from torch
def _pack_kwargs(*args, **kwargs):
    kwarg_keys = []
    flat_args = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)


def _unpack_kwargs(flat_args, kwarg_keys):
    """See _pack_kwargs."""
    assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))  # noqa: E203
    return args, kwargs
