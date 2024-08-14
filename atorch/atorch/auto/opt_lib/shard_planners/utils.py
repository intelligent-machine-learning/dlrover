"""This file provides basic utils of graph manipulation.
This includes:
    1. Turning an FX graph into a NX graph (which is more controllable)
    2. Contracting NX graph (maintaining DAG property), to reduce the size of graph.
    3. Computing the cost of an operator:
        a. Memory b. Communication c. Computation
    Some operators are distributed operators whose properties are tied with the distributed environment.
    We use DeviceTopology (describing physical connectivity of the devices) and ranks (the logical naming)
    to help pin the environment. Memory and Computation can be estimated with logical views.
    Communication are computed by first identifying the primary operators and the logical views, and then
    the actual costs are deducted from the physical view behind the logical view.
"""
import copy
import operator
from collections import deque

import networkx as nx
import torch

try:
    from pippy.IR import pipe_split
except ImportError:
    pipe_split = None

from atorch.modules.distributed_modules.mappings_registry import (
    _COMMUNICATOR_COSTS,
    get_all_gather_cost,
    get_all_to_all_cost,
    get_split_cost,
)
from atorch.modules.distributed_modules.modules_registry import (
    _BCAST_ELEMENTWISE_OPS,
    _ELEMENTWISE_OPS,
    _INTRA_OPERATOR_COMMUNICATORS,
    _MEMORY_REQUIREMENT,
    _OPERATOR_FLOPS,
    _REPLACE_SPECS,
    _RESHAPE_OPS,
    _SHARD_MAP,
    _SHARDABLE_OPERATORS,
    _SHARDED_OPERATORS,
    activation_mem_requirement,
)
from atorch.utils.graph_transform_utils import topo_sort
from atorch.utils.sharding_spec import MeshShardingSpec


class ContractableDAG(object):
    """This class is to support the node contraction for DAG.
    The initial graph must be a DAG. This class provides a method 'contract_nodes',
    by which one can contract two nodes on graph safely: i.e. the contracted graph remains to be a
    DAG.
    Loop checking is achieved with Dynamic Topological Sorting algorithm
        (a bit faster than doing full backward and forward dfs)
        https://www.doc.ic.ac.uk/~phjk/Publications/DynamicTopoSortAlg-JEA-07.pdf

    Args:
        graph (nx.DiGraph): the original DAG
        merge_info (Callable): a callable that is used to merge node info
    """

    def __init__(
        self,
        graph,
        merge_info,
    ):
        self.graph = graph

        num_nodes = len(graph.nodes)
        if len(nx.get_node_attributes(graph, "rank").values()) != num_nodes:
            # add initial ranking
            for i, node in enumerate(list(nx.topological_sort(graph))):
                graph.nodes[node]["rank"] = i
        self._visited = dict((node, False) for node in graph.nodes())
        self._merge_info = merge_info

    def remove_edge(self, source_node, target_node):
        self.graph.remove_edge(source_node, target_node)

    # this method assumes the insertion of the edge does not create loops
    def add_edge_no_check(self, source_node, target_node):
        def _get_rank(node):
            return self.graph.nodes[node]["rank"]

        def _merge_array(arr1, arr2):
            merged_arr = []
            i, j, len1, len2 = 0, 0, len(arr1), len(arr2)
            while i < len1 and j < len2:
                if arr1[i] < arr2[j]:
                    merged_arr.append(arr1[i])
                    i += 1
                else:
                    merged_arr.append(arr2[j])
                    j += 1
            while i < len1:
                merged_arr.append(arr1[i])
                i += 1
            while j < len2:
                merged_arr.append(arr2[j])
                j += 1

            return merged_arr

        source_rank = self.graph.nodes[source_node]["rank"]
        target_rank = self.graph.nodes[target_node]["rank"]
        if source_rank < target_rank:
            self.graph.add_edge(source_node, target_node)
        else:
            self._f_set = []
            self._b_set = []
            self._dfs_f(target_node, source_rank)
            self._dfs_b(source_node, target_rank)
            self._f_set.sort(key=_get_rank)
            self._b_set.sort(key=_get_rank)
            af_region = []
            # push the backward set into the affected region
            for i, node in enumerate(self._b_set):
                self._visited[node] = False
                af_region.append(node)
                # write the rank into b_set
                # avoid creating a list of rankings
                self._b_set[i] = _get_rank(node)
            for i, node in enumerate(self._f_set):
                self._visited[node] = False
                af_region.append(node)
                # write the rank into f_set
                # avoid creating a list of rankings
                self._f_set[i] = _get_rank(node)

            # merge the two rank lists
            merged_rank_list = _merge_array(self._b_set, self._f_set)
            for i in range(len(af_region)):
                node_name = af_region[i]
                self.graph.nodes[node_name]["rank"] = merged_rank_list[i]

            self.graph.add_edge(source_node, target_node)

    def _check_reachability(self, source_node, target_node):
        reachable_successors = nx.dfs_preorder_nodes(
            self.graph,
            source=source_node,
            depth_limit=self.graph.nodes[source_node]["rank"] - self.graph.nodes[source_node]["rank"],
        )
        return target_node in reachable_successors

    def _check_contrability(self, source_node, target_node):
        if self.graph.nodes[source_node]["rank"] >= self.graph.nodes[target_node]["rank"]:
            return True
        return not self._check_reachability(source_node, target_node)

    # contract source_node --> target_node
    # without checking
    def _contract_nodes(self, source_node, target_node):
        self._merge_info(source_node, target_node, self.graph)
        successors = self.graph.successors(source_node)
        predecessors = self.graph.predecessors(source_node)
        for successor in successors:
            if successor != target_node:
                self.graph.add_edge(target_node, successor)
        for predecessor in predecessors:
            if predecessor != target_node:
                self.graph.add_edge(predecessor, target_node)
        self.graph.remove_node(source_node)
        return True

    def contract_nodes(self, node_1, node_2):
        """Contract node_1 with node_2, node with higher rank is contracted into node with lower rank

        Note:
            if we contract the nodes (from target node to source node) in reversed topological orders,
            then during contraction, we are guaranteed that the target node and source node
            have not been contratced to any other nodes.
        """
        if self.graph.has_edge(node_1, node_2):
            return self.contract_edge_reversed(node_1, node_2)
        elif self.graph.has_edge(node_2, node_1):
            return self.contract_edge_reversed(node_2, node_1)
        else:
            # no edge between two nodes
            if self.graph.nodes[node_1]["rank"] < self.graph.nodes[node_2]["rank"]:
                high_node, low_node = node_1, node_2
            else:
                high_node, low_node = node_2, node_1
            # no path from low_node to high_node
            # make sure no path from high node to low node
            if self._check_reachability(high_node, low_node):
                # Do not contract
                return False
            else:
                return self._contract_nodes(high_node, low_node)

    # edge: source --> source_node
    # contraction: target_node --> edge_node
    def contract_edge_reversed(self, source_node, target_node):
        self.remove_edge(source_node, target_node)
        contractable = self._check_contrability(target_node, source_node)
        if contractable:
            self._contract_nodes(target_node, source_node)
            return True
        else:
            self.add_edge_no_check(source_node, target_node)
            return False

    # edge: source_node --> target_node
    # contraction: source_node --> target_node
    def contract_edge(self, source_node, target_node):
        self.remove_edge(source_node, target_node)
        contractable = self._check_contrability(source_node, target_node)
        if contractable:
            self._contract_nodes(source_node, target_node)
            return True
        else:
            self.add_edge_no_check(source_node, target_node)
            return False

    def _dfs_f(self, node, source_rank):
        self._f_set.append(node)
        self._visited[node] = True
        successors = self.graph.successors(node)
        for successor in successors:
            if not self._visited[successor] and self.graph.nodes[successor]["rank"] < source_rank:
                self._dfs_f(successor)
            else:
                return

    def _dfs_b(self, node, target_rank):
        self._b_set.append(node)
        self._visited[node] = True
        predecessors = self.graph.predecessors(node)
        for predecessor in predecessors:
            if not self._visited[predecessor] and self.graph.nodes[predecessor]["rank"] > target_rank:
                self._dfs_b(predecessor)
            else:
                return


# When contracting a graph, inter stage nodes cannot be merged.
# To follow the traversal order of split_module method,
# we have to loop through the graph in the insertion order
# instead of the topo order. So we have to count the partition id
# seperately.
def _mark_partition_index(fx_graph):
    part_idx = 0
    partition_map = dict()
    erase_nodes = []
    for node in fx_graph.nodes:
        if (node.op, node.target) == ("call_function", pipe_split):
            part_idx += 1
            erase_nodes.append(node)
        else:
            partition_map[node.name] = part_idx
    # clean up the graph, isolate pipe split nodes will mess up the original graph
    for node in erase_nodes:
        fx_graph.erase_node(node)
    return partition_map


def transform_graph_to_nx(fx_graph, sharding_specs, model, device_topo, prepare_merging=False):
    nx_graph = nx.DiGraph()
    partition_map = _mark_partition_index(fx_graph)
    topo_nodes = list(topo_sort(fx_graph.nodes))
    shardable_operators_names = dict((v, k) for k, v in _SHARDABLE_OPERATORS.items())
    contract_nodes = []
    ranks = device_topo.get_device_ranks()
    # insert node into the nx_graph
    # mark shardable operator, list all possible choices
    for rank, node in enumerate(topo_nodes):
        # This rank is the topo-ordering of the node
        node_attr = {
            "shardable": False,
            "sharding_spec": sharding_specs[node.name],
            "target": None,
            "orig_target": None,
            "target_name": "",
            "sharded_impls": [],
            "args": node.args,
            "part_idx": partition_map[node.name],
        }
        if prepare_merging:
            node_attr["rank"] = rank
            node_attr["aggregated_nodes"] = [node.name]
        if node.op == "call_module":
            node_target = type(model.get_submodule(node.target))
            orig_target = model.get_submodule(node.target)
        else:
            node_target = node.target
            orig_target = node.target
        node_attr["target"] = node_target
        node_attr["orig_target"] = orig_target

        target_name = shardable_operators_names.get(node_target, None)
        if target_name is None:
            # not a shardable operator, assign a name for it
            if hasattr(node_target, "__name__"):
                target_name = node_target.__name__
            else:
                target_name = str(node_target)
        node_attr["target_name"] = target_name
        shardable = target_name in _SHARDABLE_OPERATORS.keys()

        if shardable:
            node_attr["shardable"] = True
            node_attr["sharded_impls"] = []

            for sharded_impl in _SHARD_MAP[target_name]:
                if node.op == "call_module":
                    sharded_op = _SHARDED_OPERATORS[sharded_impl]
                    if sharded_op.orig_module_shardable(orig_target, ranks):
                        node_attr["sharded_impls"].append(sharded_impl)
                else:
                    node_attr["sharded_impls"].append(sharded_impl)

        if len(node_attr["sharded_impls"]) == 0:
            node_attr["shardable"] = False

        nx_graph.add_node(node.name, **node_attr)

        for input_node in list(node.all_input_nodes):
            nx_graph.add_edge(input_node.name, node.name)

        if prepare_merging:
            if node_target in _ELEMENTWISE_OPS or node_target in _RESHAPE_OPS:
                contract_nodes.append(node.name)
            elif node_target in _BCAST_ELEMENTWISE_OPS and len(list(node.all_input_nodes)) == 1:
                contract_nodes.append(node.name)
            elif node_target == operator.getitem:
                # handle the special operator __getitem__
                contract_nodes.append(node.name)

    return nx_graph, contract_nodes


def contract_nx_graph(nx_graph, contract_nodes):
    def _merge_info(source_node, target_node, graph):
        graph.nodes[target_node]["aggregated_nodes"] = (
            graph.nodes[target_node]["aggregated_nodes"] + graph.nodes[source_node]["aggregated_nodes"]
        )

    # contract_nodes in topological order
    contract_nodes.reverse()
    contracted_graph = nx_graph.copy()

    contractable_graph = ContractableDAG(contracted_graph, merge_info=_merge_info)
    for node in contract_nodes:
        input_nodes = list(contractable_graph.graph.predecessors(node))
        highest_rank = -1
        target_node = None
        for input_node in input_nodes:
            if contractable_graph.graph.nodes[input_node]["rank"] > highest_rank:
                highest_rank = contractable_graph.graph.nodes[input_node]["rank"]
                target_node = input_node
        # Contract nodes in the same stage only.
        # In theory this won't happen if we split by size as contractable nodes are of size 0
        if nx_graph.nodes[target_node]["part_idx"] == nx_graph.nodes[node]["part_idx"]:
            contractable_graph.contract_nodes(target_node, node)
    contract_nodes.reverse()
    return contractable_graph.graph


def get_intra_group_communication_cost(prev_shard, cur_shard, tensor_shape, device_topo):
    """Compute the cost of communication to reshard prev_shard to cur_shard

    FIXME only support 1 dimensional intra-group sharding now
    """
    if tensor_shape is None:
        # in shape prop, if no torch.Tensor is found, tensor_shape will be set to None
        return 0
    if isinstance(prev_shard, (list, tuple)):
        total_cost = sum(
            [
                get_intra_group_communication_cost(p_shard, c_shard, t_shape, device_topo)
                for p_shard, c_shard, t_shape in zip(prev_shard, cur_shard, tensor_shape)
            ]
        )
        return total_cost
    elif isinstance(prev_shard, dict):
        total_cost = sum(
            [
                get_intra_group_communication_cost(p_shard, c_shard, t_shape, device_topo)
                for p_shard, c_shard, t_shape in zip(prev_shard, cur_shard, tensor_shape)
            ]
        )
        return total_cost
    elif isinstance(prev_shard, slice):
        # a slice object cannot be sharded, return 0
        return 0
    else:
        # the output of previous node is not a tensor, do not shard it
        if prev_shard is None or prev_shard == cur_shard:
            return 0

        if len(prev_shard.dims) > 1 or len(cur_shard.dims) > 1:
            raise ValueError(f"Only 1 dimension sharding is supported, cannot handle {prev_shard.dims}")

        subset_topo = device_topo.get_physical_topology(prev_shard.ranks)
        # forward + backward
        if len(prev_shard.dims) == 0 or len(cur_shard.dims) == 0:
            return get_split_cost(subset_topo, tensor_shape) + get_all_gather_cost(subset_topo, tensor_shape)
        else:
            return 2 * get_all_to_all_cost(subset_topo, tensor_shape)


def propagate_through_subgraph(node, new_sharding_spec, sharding_specs, contracted_graph, nx_graph):
    """Since mergable nodes are contracted to predecessors,
    Any nodes that needs a decision is a root node of the subgraph by its aggregated nodes.
    This method propagate the sharding spec through the nx_graph and rewrites the sharding specs
    of aggregated nodes of the node.

    Other than the root node, all other merged nodes has only one incoming edge.

    during propagation, reshape_ops are always assumed to be fully replicated, for other nodes,
    the sharding_spec are aligned with the root node's sharding spec.

    Args:
        node: the root node of the contracted node
        new_sharding_spec: the new sharding spec of the root node
        sharding_specs: the dict of sharding specs of the whole graph
        contracted_graph: the contracted graph
        nx_graph: the original graph
    """
    agg_nodes = contracted_graph.nodes[node]["aggregated_nodes"]
    prop_spec = dict((k, sharding_specs[k]) for k in sharding_specs.keys() if k in agg_nodes and k != node)

    if new_sharding_spec == sharding_specs[node]["output_spec"]:
        return prop_spec

    new_sharding_spec_cp = copy.deepcopy(new_sharding_spec)
    prop_spec_cp = copy.deepcopy(prop_spec)
    visited = dict((node, False) for node in agg_nodes)
    queue = deque()
    queue.append(node)
    while len(queue) > 0:
        visit_node = queue.popleft()
        node_target = nx_graph.nodes[visit_node]["target"]
        if not visited[visit_node]:
            for user_node in nx_graph.successors(visit_node):
                if user_node in agg_nodes and not visited[user_node]:
                    queue.append(user_node)
        if visit_node == node:
            continue

        if node_target not in _RESHAPE_OPS:
            # all merged nodes has only one input
            predecessor = list(nx_graph.predecessors(visit_node))[0]
            predecessor_output_spec = (
                new_sharding_spec_cp if predecessor == node else prop_spec_cp[predecessor]["output_spec"]
            )
            prop_spec_cp[visit_node]["input_spec"][predecessor] = predecessor_output_spec
            prop_spec_cp[visit_node]["output_spec"] = (
                predecessor_output_spec
                if node_target != operator.getitem or isinstance(predecessor_output_spec, MeshShardingSpec)
                else operator.getitem(predecessor_output_spec, nx_graph.nodes[visit_node]["args"][-1])
            )
        visited[visit_node] = True

    return prop_spec_cp


def generate_intra_node_com_cost(node, choice, sharding_specs, tensor_shapes, device_topo, contracted_graph, nx_graph):
    """Propogate the new sharding spec within the contracted node, and generate the intra contracted node
    communication cost

    Args:
        node: the root node of the contracted node
        sharding_specs: the dict of sharding specs of the whole graph
        contracted_graph: the contracted graph
        nx_graph: the original graph
        tensor_shapes: shapes of node outputs
        choice: the distributed implementation chosen for the root node
        device_topo: the device topology
    """
    if choice == 0:
        return 0
    impl_name = contracted_graph.nodes[node]["sharded_impls"][choice - 1]
    node_output_spec = copy.deepcopy(sharding_specs[node]["output_spec"])
    ranks = device_topo.get_device_ranks()
    _, node_output_spec = _REPLACE_SPECS[impl_name](MeshShardingSpec(()), node_output_spec, ranks=ranks)
    prop_spec = propagate_through_subgraph(node, node_output_spec, sharding_specs, contracted_graph, nx_graph)
    agg_nodes = set(contracted_graph.nodes[node]["aggregated_nodes"])
    super_node_subgraph = nx_graph.subgraph(agg_nodes)
    intra_node_com_cost = 0
    for (node_i, node_j) in super_node_subgraph.edges():
        prev_shard = node_output_spec if node_i == node else prop_spec[node_i]["output_spec"]
        cur_shard = prop_spec[node_j]["input_spec"][node_i]
        tensor_shape = tensor_shapes[node_i]
        edge_cost = get_intra_group_communication_cost(prev_shard, cur_shard, tensor_shape, device_topo)
        intra_node_com_cost += edge_cost

    # no resharding needed for nodes other than the root node
    node_target = contracted_graph.nodes[node]["orig_target"]
    ranks = device_topo.get_device_ranks()
    intra_op_communicators = _INTRA_OPERATOR_COMMUNICATORS.get(impl_name, lambda ranks, tensor_shape, node_target: [])(
        ranks, tensor_shapes[node], node_target
    )
    node_com_cost = sum(
        [
            _COMMUNICATOR_COSTS.get(communicator_name, lambda subset_topo, tensor_shape: 0)(
                device_topo.get_physical_topology(op_ranks), op_tensor_shape
            )
            for (communicator_name, op_ranks, op_tensor_shape) in intra_op_communicators
        ]
    )
    return intra_node_com_cost + node_com_cost


def _get_optimizer_state_num(optimizer):
    # TODO handle more types of optimizers
    if optimizer is not None:
        if isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam)):
            # first and second momentum
            optimizer_mem_factor = 2
        else:
            optimizer_mem_factor = 1
    else:
        optimizer_mem_factor = 1
    return optimizer_mem_factor


def _generate_node_memory_requirement(
    node,
    choice,
    tensor_shapes,
    ranks,
    contracted_graph,
    nx_graph,
    optimizer=None,
    forward_factor=1.0,
    grad_factor=1.0,
    param_factor=1.0,
    optimizer_factor=1.0,
):
    """Generate the memory requirement of a node"""
    orig_target = nx_graph.nodes[node]["orig_target"]
    tensor_shape = tensor_shapes[node]
    if choice == 0:
        impl_name = nx_graph.nodes[node]["target_name"]
    else:
        impl_name = contracted_graph.nodes[node]["sharded_impls"][choice - 1]
    forward_mem, grad_mem, params_mem = _MEMORY_REQUIREMENT.get(impl_name, activation_mem_requirement)(
        ranks, tensor_shape, orig_target
    )
    optimizer_mem_factor = _get_optimizer_state_num(optimizer)
    # add 0.2 to avid cuda oom
    total_mem = (
        forward_factor * forward_mem
        + grad_factor * grad_mem
        + param_factor * (optimizer_factor * optimizer_mem_factor + 0.2 + 1) * params_mem
    )
    return total_mem


def generate_node_memory_requirement(
    node,
    choice,
    tensor_shapes,
    ranks,
    contracted_graph,
    nx_graph,
    optimizer=None,
    forward_factor=1.0,
    grad_factor=1.0,
    param_factor=1.0,
    optimizer_factor=1.0,
):
    agg_nodes = contracted_graph.nodes[node]["aggregated_nodes"]
    total_mem = 0
    for agg_node in agg_nodes:
        cur_choice = choice if agg_node == node else 0
        total_mem += _generate_node_memory_requirement(
            agg_node,
            cur_choice,
            tensor_shapes,
            ranks,
            contracted_graph,
            nx_graph,
            optimizer,
            forward_factor,
            grad_factor,
            param_factor,
            optimizer_factor,
        )
    return total_mem


def _generate_node_flops(node, choice, tensor_shapes, ranks, contracted_graph, nx_graph):
    """Generate the flops of a node, aggregated nodes and unrecognized nodes are ignored"""
    orig_target = nx_graph.nodes[node]["orig_target"]
    tensor_shape = tensor_shapes[node]
    if choice == 0:
        impl_name = nx_graph.nodes[node]["target_name"]
    else:
        impl_name = contracted_graph.nodes[node]["sharded_impls"][choice - 1]
    node_flops = _OPERATOR_FLOPS.get(impl_name, lambda anks, tensor_shape, orig_module: 0)(
        ranks, tensor_shape, orig_target
    )
    return node_flops


def generate_node_flops(node, choice, tensor_shapes, ranks, contracted_graph, nx_graph):
    agg_nodes = contracted_graph.nodes[node]["aggregated_nodes"]
    total_flops = 0
    for agg_node in agg_nodes:
        if agg_node == node:
            total_flops += _generate_node_flops(node, choice, tensor_shapes, ranks, contracted_graph, nx_graph)
        else:
            total_flops += _generate_node_flops(agg_node, 0, tensor_shapes, ranks, contracted_graph, nx_graph)
    return total_flops


def generate_inter_nodes_com_cost(
    node_i, node_j, choice_i, choice_j, sharding_specs, tensor_shapes, device_topo, contracted_graph, nx_graph
):
    """Find all cut edges between two contracted nodes. Since there is no loop in the contracted graph,
    all cut edges must be from node_i to node_j.

    Args:
        node_i: the source node
        node_j: the target node
        choice_i: the distributed implementation chosen for node_i
        choice_j: the distributed implementation chosen for node_j
        sharding_specs: the dict of sharding specs of the whole graph
        device_topo: the device topology on which tensor parallelism is operated
        tensor_shapes: shape of node outputs
        contracted_graph: the contracted graph
        nx_graph: the original nx_graph
    """
    total_edge_cost = 0
    input_shard_i, output_shard_i = copy.deepcopy(sharding_specs[node_i]["input_spec"]), copy.deepcopy(
        sharding_specs[node_i]["output_spec"]
    )
    input_shard_j, output_shard_j = copy.deepcopy(sharding_specs[node_j]["input_spec"]), copy.deepcopy(
        sharding_specs[node_j]["output_spec"]
    )
    ranks = device_topo.get_device_ranks()
    if choice_i != 0:
        impl_name_i = contracted_graph.nodes[node_i]["sharded_impls"][choice_i - 1]
        input_shard_i, output_shard_i = _REPLACE_SPECS[impl_name_i](input_shard_i, output_shard_i, ranks=ranks)

    if choice_j != 0:
        impl_name_j = contracted_graph.nodes[node_j]["sharded_impls"][choice_j - 1]
        input_shard_j, output_shard_j = _REPLACE_SPECS[impl_name_j](input_shard_j, output_shard_j, ranks=ranks)
    cut_edges = nx.edge_boundary(
        nx_graph, contracted_graph.nodes[node_i]["aggregated_nodes"], contracted_graph.nodes[node_j]["aggregated_nodes"]
    )
    prop_spec_i = propagate_through_subgraph(node_i, output_shard_i, sharding_specs, contracted_graph, nx_graph)
    prop_spec_j = propagate_through_subgraph(node_j, output_shard_j, sharding_specs, contracted_graph, nx_graph)
    prop_spec_i[node_i] = {"input_spec": input_shard_i, "output_spec": output_shard_i}
    prop_spec_j[node_j] = {"input_spec": input_shard_j, "output_spec": output_shard_j}

    for source_node, target_node in cut_edges:
        output_shard = prop_spec_i[source_node]["output_spec"]
        input_shard = prop_spec_j[target_node]["input_spec"][source_node]
        tensor_shape = tensor_shapes[source_node]
        edge_cost = get_intra_group_communication_cost(output_shard, input_shard, tensor_shape, device_topo)
        total_edge_cost += edge_cost
    return total_edge_cost
