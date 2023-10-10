""""fx.Graph -> networkx graph -> MIP
FIXME fuse local/non-distributed operators into distributed operators,
to reduce the number of decision variables
"""
import copy
import itertools
import math
import os
import time

import numpy as np
import pyomo.environ as pyo

from atorch.auto.opt_lib.shard_planners.base_tp_planner import BaseTensorParallelPlanner
from atorch.auto.opt_lib.utils import print_sharding_specs
from atorch.common.log_utils import default_logger as logger
from atorch.modules.distributed_modules.modules_registry import _REPLACE_SPECS

from .utils import (
    contract_nx_graph,
    generate_inter_nodes_com_cost,
    generate_intra_node_com_cost,
    generate_node_flops,
    generate_node_memory_requirement,
    propagate_through_subgraph,
    transform_graph_to_nx,
)


class MIPTensorParallelPlanner(BaseTensorParallelPlanner):
    """An MIP tensor parallel shard planner.
    This planner detects the stage split points in the graph and construct the cost/constraint model
    seperately for each stage.
    """

    def __init__(
        self,
        memory_bound=1,
        fp32_flops=1,
        merge_nodes=True,
        strategy_name="mip_strategy",
        solver="glpk",
        timelimit=600,
        greedy_init=False,
    ):
        self.strategy_name = strategy_name
        self.memory_bound = memory_bound
        self.fp32_flops = fp32_flops
        self.solver = solver
        self.timelimit = timelimit
        self.merge_nodes = merge_nodes
        self.greedy_init = greedy_init
        self.forward_factor = 1.0
        self.grad_factor = 1.0
        self.param_factor = 1.0
        self.optimizer_factor = 1.0

    def set_mem_factors(self, forward_factor, grad_factor, param_factor, optimizer_factor):
        self.forward_factor = forward_factor
        self.grad_factor = grad_factor
        self.param_factor = param_factor
        self.optimizer_factor = optimizer_factor

    def _get_mip_parameters(self, model, graph, sharding_specs, tensor_shapes, device_topo, optimizer=None):
        """construct mip model parameters
        FIXME This method depends on the ability to assess the memory/communication cost
        of each operator/parallel operator. Refactor this method into a clearer form
        FIXME Maybe some sort of interpretor to do this
        Args:
            model: original model
            graph: fx graph
            sharding_specs: initial sharding specs
            tensor_shapes: the extracted tensor shapes of each node's output,
                this is supposed to be used for inferring communication cost and memory requirement
                to be refactored
            device_topo: the physical topology of devices on which to distribute the model
                These devices are symoblic, corresponding to the process_group created for tensor
                parallelism

        model variables:
            x_{i, j}: decision variable for shardable operator i, whether to use impl j
                j correspond to distributed implementation graph[node_name]['sharded_impl'][j - 1]
                j = 0 correspond to the original implementation
            e_{i, k, j, r}: decision variable for edge i, j, to linearize the cost.
                e_{i, k, j, r}  = 1 -> node i chooses impl k, node j chooses impl r

        Return:
            inter_operator_communication_costs: communication cost of each edge
                keys: ((i, k), (j, r)), value: if node i is parallelized as k, j is parallelized
                as r, then value is the communication cost of resharding
            intra_operator_communication_costs: communication cost of each operator
                keys: (i, k), value: if node i is parallelized as k, then the value is the communication
                cost of the corresponding parallel operator
            memory_requirements: memory requirement of each operator
                keys: (i, k), value: if node i is parallelized as k, then the value is the memory
                requirement of this parallel operator
            contracted_graph: contracted graph
            nx_graph: networkx graph that is not contracted
        """
        # separate coefficient generation and model construction
        nx_graph, contract_nodes = transform_graph_to_nx(graph, sharding_specs, model, device_topo, True)
        contracted_graph = contract_nx_graph(nx_graph, contract_nodes) if self.merge_nodes else nx_graph
        # FIXME use a more efficient data structure to store the MIP parameters
        memory_requirements = dict()
        operator_flops = dict()

        intra_node_communication_costs = dict()
        nodes = list(contracted_graph.nodes())
        for i, node in enumerate(nodes):
            num_choices = len(contracted_graph.nodes[node]["sharded_impls"])
            for k in range(num_choices + 1):
                com_cost = generate_intra_node_com_cost(
                    node, k, sharding_specs, tensor_shapes, device_topo, contracted_graph, nx_graph
                )
                intra_node_communication_costs[(i, k)] = com_cost
                mem_req = generate_node_memory_requirement(
                    node,
                    k,
                    tensor_shapes,
                    device_topo.get_device_ranks(),
                    contracted_graph,
                    nx_graph,
                    optimizer,
                    self.forward_factor,
                    self.grad_factor,
                    self.param_factor,
                    self.optimizer_factor,
                )
                memory_requirements[(i, k)] = mem_req
                flops = generate_node_flops(
                    node, k, tensor_shapes, device_topo.get_device_ranks(), contracted_graph, nx_graph
                )
                operator_flops[(i, k)] = flops

        inter_nodes_communication_costs = dict()

        for (node_i, node_j) in contracted_graph.edges():
            i, j = nodes.index(node_i), nodes.index(node_j)
            num_choices_i = len(contracted_graph.nodes[node_i]["sharded_impls"])
            num_choices_j = len(contracted_graph.nodes[node_j]["sharded_impls"])
            for (k, r) in itertools.product(list(range(num_choices_i + 1)), list(range(num_choices_j + 1))):

                edge_cost = generate_inter_nodes_com_cost(
                    node_i, node_j, k, r, sharding_specs, tensor_shapes, device_topo, contracted_graph, nx_graph
                )
                inter_nodes_communication_costs[((i, j), (k, r))] = edge_cost

        return (
            intra_node_communication_costs,
            inter_nodes_communication_costs,
            memory_requirements,
            operator_flops,
            contracted_graph,
            nx_graph,
        )

    def _greedy_initialization(
        self,
        contracted_graph,
        memory_requirements,
    ):
        """Compute an initial assignment greedily. The target is to fullfil the memory requirement"""
        total_memory_req = 0
        stage_nodes = dict()
        node_to_node_index = dict()
        for idx, node in enumerate(contracted_graph.nodes()):
            node_to_node_index[node] = idx
            stage_nodes.setdefault(contracted_graph.nodes[node]["part_idx"], []).append(node)

        initial_assignment = dict()
        stage_total_memory_reqs = dict()
        for part_idx, nodes in stage_nodes.items():
            total_memory_req = 0
            for idx, node in enumerate(nodes):
                node_idx = node_to_node_index[node]
                total_memory_req += memory_requirements[(node_idx, 0)]
                initial_assignment[node_idx] = 0
            idx = 0
            while total_memory_req > self.memory_bound and idx < len(nodes):
                # loop over mem_req
                node = nodes[idx]
                node_idx = node_to_node_index[node]
                min_req, min_index = memory_requirements[(node_idx, 0)], 0
                for k in range(len(contracted_graph.nodes[node]["sharded_impls"]) + 1):
                    if memory_requirements[(node_idx, k)] < min_req:
                        min_index, min_req = k, memory_requirements[(node_idx, k)]
                initial_assignment[node_idx] = min_index
                total_memory_req -= memory_requirements[(node_idx, 0)] - min_req
                idx += 1

            stage_total_memory_reqs[part_idx] = total_memory_req

        return initial_assignment, stage_total_memory_reqs

    def _construct_mip_model(
        self,
        contracted_graph,
        intra_node_communication_costs,
        inter_nodes_communication_costs,
        memory_requirements,
        operator_flops,
        device_topo,
        fp32_flops,
        greedy_init=None,
        flops_cost_factor=None,
    ):
        """Given the parameters, construct an mip model
        model variables:
           x_{i, j}: decision variable for shardable operator i, whether to use impl j
               j correspond to distributed implementation graph[node_name]['sharded_impl'][j - 1]
               j = 0 correspond to the original implementation
           e_{i, k, j, r}: decision variable for edge i, j, to linearize the cost.
               e_{i, k, j, r}  = 1 -> node i chooses impl k, node j chooses impl r
        """
        # FIXME loop through the graph to get the number of stages, inefficient
        nodes = list(contracted_graph.nodes())
        nstages = len(set(contracted_graph.nodes[node]["part_idx"] for node in nodes))
        node_vars = intra_node_communication_costs.keys()
        edge_vars = inter_nodes_communication_costs.keys()
        model = pyo.ConcreteModel()
        model.constraints = pyo.ConstraintList()

        stage_total_memory_reqs = None

        greedy_init = self.greedy_init if greedy_init is None else greedy_init
        if greedy_init:
            initial_assignment, stage_total_memory_reqs = self._greedy_initialization(
                contracted_graph, memory_requirements
            )
        else:
            initial_assignment = {i: 0 for i in range(len(contracted_graph.nodes()))}

        if stage_total_memory_reqs is not None and min(stage_total_memory_reqs.values()) > self.memory_bound:
            # if even greedy initialization does not fit, drop and return the best possible assignment
            return initial_assignment, stage_total_memory_reqs

        # Initialization: assumes all nodes are not partitioned
        model.x = pyo.Var(node_vars, within=pyo.Binary, initialize=lambda model, i, k: int(initial_assignment[i] == k))
        model.e = pyo.Var(
            edge_vars,
            within=pyo.Binary,
            initialize=lambda model, i, j, k, r: int(initial_assignment[i] == k and initial_assignment[j] == r),
        )

        # decision constraint
        for i, node in enumerate(nodes):
            num_choices = len(contracted_graph.nodes[node]["sharded_impls"])
            model.constraints.add(sum(model.x[(i, k)] for k in range(num_choices + 1)) == 1)
        # edge constraint
        for ((i, j), (k, r)) in edge_vars:
            model.constraints.add(model.e[((i, j), (k, r))] - model.x[(i, k)] <= 0)
            model.constraints.add(model.e[((i, j), (k, r))] - model.x[(j, r)] <= 0)
            model.constraints.add(model.x[(i, k)] + model.x[(j, r)] - model.e[((i, j), (k, r))] <= 1)
        # memory constraint for each stage
        for part_idx in range(nstages):
            model.constraints.add(
                sum(
                    mem_req * model.x[(i, k)]
                    for (i, k), mem_req in memory_requirements.items()
                    if contracted_graph.nodes[nodes[i]]["part_idx"] == part_idx
                )
                <= self.memory_bound
            )

        # objective
        if flops_cost_factor is None:
            device_world_size = device_topo.num_devices()
            default_flops_cost_factor = 4 * math.sqrt(device_world_size) * device_world_size
            flops_cost_factor = os.getenv("FLOPS_COST_FACTOR", default_flops_cost_factor)
        model.com_cost = pyo.Objective(
            expr=(
                sum(
                    inter_nodes_communication_costs[((i, j), (k, r))] * model.e[((i, j), (k, r))]
                    for ((i, j), (k, r)) in edge_vars
                )
                + sum(intra_node_communication_costs[(i, k)] * model.x[(i, k)] for (i, k) in node_vars)
            )
            + flops_cost_factor * sum(flops * model.x[(i, k)] for (i, k), flops in operator_flops.items()) / fp32_flops,
            sense=pyo.minimize,
        )

        return model, stage_total_memory_reqs

    def fake_profile(self, model, graph, sharding_specs, tensor_shapes, device_topo, fp32_flops, optimizer=None):
        """Given the device topology, the graph, this method offers
        to profile the graph, giving a rough estimate of the runtime of each node (total flops + total btyes comm),
        the communicate time, on node memory requirements.

        A more accurate method would be to specify the stage partition first then run an estimation for each individual
        partition, but this is too costly.

        With all the info, we ca solve an optimal chunks. Currently assume no interleaved schedule employed.
        Even if checkpointing is used, in the backward pass all forward activations must be stored for a single
        chunk, so we give the full estimation and leave the post processing for scaling the estimate

        Args:
            model: original model
            graph: fx.Graph
            sharding_specs: sharding specs of the model
            tensor_shapes: tensor shapes of the output of each node. Supposedly to be used
                for inferring memory and communication cost. to be refactored.
            device_topo: the physical topology of devices on which to distribute the model
                These devices are symoblic, corresponding to the process_group created for tensor
                parallelism
        """
        (
            intra_node_communication_costs,
            inter_nodes_communication_costs,
            memory_requirements,
            operator_flops,
            contracted_graph,
            _,
        ) = self._get_mip_parameters(model, graph, sharding_specs, tensor_shapes, device_topo, optimizer)
        device_world_size = device_topo.num_devices()
        flops_cost_factor = 4 * device_world_size**2
        # TODO we are enforcing the MIP to solve a model without greedy init for stability
        mip_model, _ = self._construct_mip_model(
            contracted_graph,
            intra_node_communication_costs,
            inter_nodes_communication_costs,
            memory_requirements,
            operator_flops,
            device_topo,
            self.fp32_flops,
            greedy_init=False,
            flops_cost_factor=flops_cost_factor,
        )
        nodes = list(contracted_graph.nodes())
        node_vars = intra_node_communication_costs.keys()
        edge_vars = inter_nodes_communication_costs.keys()
        estimated_node_memory_reqs = np.zeros(len(nodes), dtype=float)
        estimated_node_flops_cost = np.zeros(len(nodes), dtype=float)
        estimated_intra_node_comm_cost = np.zeros(len(nodes), dtype=float)
        estimated_edge_comm_cost = np.zeros((len(nodes), len(nodes)), dtype=float)

        # Solve the MIP model
        solver = pyo.SolverFactory(self.solver)
        if self.solver == "glpk":
            # set a tighter time limit
            solver.options["tmlim"] = 60

        solver.solve(mip_model)
        for (i, k) in node_vars:
            estimated_node_memory_reqs[i] += memory_requirements[(i, k)] * mip_model.x[(i, k)].value
            estimated_node_flops_cost[i] += operator_flops[(i, k)] * mip_model.x[(i, k)].value / fp32_flops
            estimated_intra_node_comm_cost[i] += intra_node_communication_costs[(i, k)] * mip_model.x[(i, k)].value

        # FIXME the inter_nodes_communication_costs is the cost for resharding
        # what we need is to compute the activation size
        for ((i, j), (k, r)) in edge_vars:
            estimated_edge_comm_cost[i][j] += (
                inter_nodes_communication_costs[((i, j), (k, r))] * mip_model.e[((i, j), (k, r))].value
            )

        return (
            estimated_node_memory_reqs,
            estimated_node_flops_cost,
            estimated_intra_node_comm_cost,
            estimated_edge_comm_cost,
            contracted_graph,
        )

    def generate_sharding_plan(self, model, graph, sharding_specs, tensor_shapes, device_topo, optimizer=None):
        """Generate a sharding plan with MIP

        Args:
            model: original model
            graph: fx.Graph
            sharding_specs: sharding specs of the model
            tensor_shapes: tensor shapes of the output of each node. Supposedly to be used
                for inferring memory and communication cost. to be refactored.
            device_topo: the physical topology of devices on which to distribute the model
                These devices are symoblic, corresponding to the process_group created for tensor
                parallelism
        """
        replaced_specs = copy.deepcopy(sharding_specs)
        (
            intra_node_communication_costs,
            inter_nodes_communication_costs,
            memory_requirements,
            operator_flops,
            contracted_graph,
            nx_graph,
        ) = self._get_mip_parameters(model, graph, sharding_specs, tensor_shapes, device_topo, optimizer)
        mip_model, stage_total_memory_reqs = self._construct_mip_model(
            contracted_graph,
            intra_node_communication_costs,
            inter_nodes_communication_costs,
            memory_requirements,
            operator_flops,
            device_topo,
            self.fp32_flops,
        )

        shardable_nodes = []
        sharded_nodes = []
        for node in contracted_graph.nodes():
            if contracted_graph.nodes[node]["shardable"]:
                shardable_nodes.append(node)

        replacement_map = dict()
        process_group_assignment = dict()

        nodes = list(contracted_graph.nodes())
        nstages = len(list(stage_total_memory_reqs.keys()))
        estimated_stage_memory_reqs = dict()
        tp_debug = logger.root.level > 30
        if stage_total_memory_reqs is not None and min(stage_total_memory_reqs.values()) > self.memory_bound:
            if tp_debug:
                logger.info("[TP DEBUG] Greedy Assignment for TP")
            estimated_stage_memory_reqs = stage_total_memory_reqs
            # mip_model is exactly the assignment
            for i, node in enumerate(nodes):
                k = mip_model[i]
                if k != 0:
                    sharded_nodes.append(node)
                    replaced_target = contracted_graph.nodes[node]["sharded_impls"][k - 1]
                    replacement_map[node] = replaced_target
                    process_group_assignment[node] = dict()
                    process_group_assignment[node]["group"] = "tensor"
                    process_group_assignment[node]["ranks"] = None

        else:
            num_decision_vars = len(list(memory_requirements.keys())) + len(
                list(inter_nodes_communication_costs.keys())
            )
            logger.info(f"There are {num_decision_vars} decision variables")
            solver = pyo.SolverFactory(self.solver)
            if self.timelimit is not None and self.solver == "glpk":
                # timelimit is only tested for glpk solver
                solver.options["tmlim"] = self.timelimit
            logger.info("start solving for a plan")
            start_time = time.time()
            results = solver.solve(mip_model)
            optimality = results.Solver[0]["Termination condition"]
            logger.info(f"the final result is {optimality}")
            elapsed_time = time.time() - start_time
            logger.info(f"Solving for a sharding plan took {elapsed_time / 60} mins")

            # parse the result
            for (i, k) in memory_requirements.keys():
                node = nodes[i]
                node_decision = mip_model.x[(i, k)].value
                if node_decision != 0 and k != 0:
                    sharded_nodes.append(node)
                    replaced_target = contracted_graph.nodes[node]["sharded_impls"][k - 1]
                    # assigne the node to group "tensor"
                    replacement_map[node] = replaced_target
                    process_group_assignment[node] = dict()
                    process_group_assignment[node]["group"] = "tensor"
                    process_group_assignment[node]["ranks"] = None
            for part_idx in range(nstages):
                estimated_stage_memory_reqs[part_idx] = sum(
                    mem_req * mip_model.x[(i, k)].value
                    for (i, k), mem_req in memory_requirements.items()
                    if contracted_graph.nodes[nodes[i]]["part_idx"] == part_idx
                )

        frac_sharded_nodes = len(sharded_nodes) / len(shardable_nodes)
        logger.info(f"{frac_sharded_nodes} of shardable nodes are sharded")
        changed_local_nodes = set()
        for node in replacement_map.keys():
            (replaced_specs[node]["input_spec"], replaced_specs[node]["output_spec"]) = _REPLACE_SPECS[
                replacement_map[node]
            ](
                replaced_specs[node]["input_spec"],
                replaced_specs[node]["output_spec"],
                group=process_group_assignment[node]["group"],
                ranks=process_group_assignment[node]["ranks"],
            )
            prop_spec = propagate_through_subgraph(
                node, replaced_specs[node]["output_spec"], replaced_specs, contracted_graph, nx_graph
            )
            if tp_debug:
                output_spec = replaced_specs[node]["output_spec"]
                logger.info(f"[TP_DEBUG] Propagated specs for root {node} with output spec: {str(output_spec)}")
                print_sharding_specs(prop_spec)

            replaced_specs.update(prop_spec)
            changed_local_nodes.update(list(prop_spec.keys()))

        logger.info(f"replacement_map: {replacement_map}")
        for part_idx in estimated_stage_memory_reqs.keys():
            estimated_stage_memory_reqs[part_idx] /= 1073741824
        logger.info(f"memory bound: {self.memory_bound / 1073741824}, estimated_mem_req: {estimated_stage_memory_reqs}")

        # FIXME replaced_specs/process_group_assignment may contain ProcessGroup object, which cannot be serialized
        best_config = {
            "replacement_map": replacement_map,
            "replaced_specs": replaced_specs,
            "changed_local_nodes": changed_local_nodes,
            "process_group_assignment": process_group_assignment,
            "process_groups": dict(),
        }

        return best_config
