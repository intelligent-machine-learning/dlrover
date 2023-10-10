# Another MIP based method to solve for some critical parameters of Mixed Parallel
# To fire off a mixed parallel model we need to decide:
# GPU partition: data size, pipe size, tensor size
# Pipeline: partition; chunks (micro-batchsize);
# Tensor partition
import copy

import numpy as np

from atorch.auto.opt_lib.optimization import DistributedGraphMixin
from atorch.distributed.distributed import _get_pg_ranks

from .mip_tp_planner import MIPTensorParallelPlanner


# FIXME estimated_edge_comm_cost is computed with the TP group device topo, it should be scaled here
def tera_pipe_partition(
    num_nodes,
    estimated_node_memory_reqs,
    estimated_node_flops_cost,
    estimated_intra_node_comm_cost,
    estimated_edge_comm_cost,
    num_devices,
    memory_limit,
    gap_value=0.1,
):
    """
    Implement modularized TeraPipe-based partitioning algorithm with a value-based gap for Tmax.
    """
    intra_costs = estimated_node_flops_cost + estimated_intra_node_comm_cost
    cumsum_intra_cost = np.cumsum(intra_costs)
    cumsum_mem_reqs = np.cumsum(estimated_node_memory_reqs)

    # Initialize an empty 2D array of the same size
    stage_intra_costs = np.zeros((len(intra_costs), len(intra_costs)))
    stage_mem_reqs = np.zeros((len(estimated_node_memory_reqs), len(estimated_node_memory_reqs)))

    # Fill the 2D array with the sums
    for i in range(len(intra_costs)):
        for k in range(i, len(intra_costs)):
            stage_intra_costs[i, k] = cumsum_intra_cost[k] - (cumsum_intra_cost[i - 1] if i > 0 else 0)
            stage_mem_reqs[i, k] = cumsum_mem_reqs[k] - (cumsum_mem_reqs[i - 1] if i > 0 else 0)

    sorted_intra_costs = sorted(np.unique(stage_intra_costs))

    # List to store the DP values
    f = {}

    # Dictionary to store the optimal partition points
    partition_points = {}

    # Initialize the optimal forward cost
    min_cost = float("inf")

    # Initialize previous Tmax to 0
    prev_Tmax = 0

    best_Tmax = None  # Ensure best_Tmax is initialized

    # Loop over possible values of Tmax using sorted intra_costs and gap_value
    for Tmax in sorted_intra_costs:
        if Tmax < prev_Tmax + gap_value:
            continue
        if (num_devices - 1) * Tmax >= min_cost:
            break

        f[Tmax] = [[float("inf")] * (num_devices + 1) for _ in range(num_nodes + 1)]
        f[Tmax][0][0] = 0  # Base case

        for i in range(1, num_nodes + 1):
            for s in range(1, num_devices + 1):  # Loop over number of stages
                for k in range(i):
                    cost = stage_intra_costs[k, i - 1]
                    mem = stage_mem_reqs[k, i - 1]
                    partition_cost = (
                        f[Tmax][k][s - 1]
                        + cost
                        + sum(estimated_edge_comm_cost[r][t] for r, t in zip(range(k + 1), range(k + 1, i)))
                    )
                    if cost <= Tmax and mem <= memory_limit:
                        if partition_cost < f[Tmax][i][s]:
                            f[Tmax][i][s] = partition_cost
                            partition_points[(i, s, Tmax)] = k

        cost = (num_devices - 1) * Tmax + f[Tmax][-1][-1]
        if cost < min_cost:
            min_cost = cost
            best_Tmax = Tmax
        prev_Tmax = Tmax

    if best_Tmax is None:
        return None

    optimal_partitions = []
    i = num_nodes
    s = num_devices
    while i > 0 and s > 0:
        k = partition_points[(i, s, best_Tmax)]
        optimal_partitions.append((k, i))
        i = k
        s -= 1

    return list(reversed(optimal_partitions))


class DimPlanner(DistributedGraphMixin):
    def __init__(
        self,
        num_nodes=None,
        num_devices_per_node=None,
        tracer_backend: str = "meta_fx",
        prop_mode: str = "interpreter",
        use_fake_mode: bool = False,
        device_context=None,
    ):
        super().__init__(
            num_nodes=num_nodes,
            num_devices_per_node=num_devices_per_node,
            tracer_backend=tracer_backend,
            prop_mode=prop_mode,
            use_fake_mode=use_fake_mode,
            device_context=device_context,
        )
        self.profiler = MIPTensorParallelPlanner(
            memory_bound=self.memory_bound,
            fp32_flops=self.fp32_flops,
            merge_nodes=True,
            solver="glpk",
            greedy_init=True,
            timelimit=120,
        )

    def generate_sharding_plan(self, model_context, config=dict()):
        gap_value = config.get("gap_value", 0)
        # deepcopy model_context
        mc = copy.deepcopy(model_context)
        mc.convert_to_loss_wrapper()
        optimizer = mc.create_optim()
        total_devices = self.num_nodes * self.num_devices_per_node

        tensor_sizes = [2**i for i in range(1, int(np.log2(self.num_devices_per_node)) + 1)]

        # Calculate all possible combinations of tensor and pipe sizes, sorted by their product
        combinations = [
            (t, p) for t in tensor_sizes for p in range(1, total_devices // t + 1) if total_devices % (t * p) == 0
        ]
        combinations.sort(key=lambda x: x[0] * x[1])

        # TODO
        # We are making some simplifying assumption: we assume chunks = num_stages
        # Assuming current impl of StageInterleaver, In steady state,
        # there will be chunks - num_stages/2 = num_stages/2 chunks
        # This simplifies our fake profiling

        # This search loop
        optimal_tensor_size = None
        optimal_pipe_size = None
        optimal_data_size = None
        optimal_partitions = None
        optimal_product = float("inf")

        for tensor_size, pipe_size in combinations:

            current_product = tensor_size * pipe_size

            # If we've already found an optimal partition and current product is larger, break
            if optimal_partitions and current_product > optimal_product:
                break

            data_size = total_devices // current_product
            parallel_config = {"ddp_size": data_size, "chunks": pipe_size}
            graph, sharding_specs, tensor_shapes = self._trace_and_propagate(
                mc, config, strategy=None, parallel_config=parallel_config
            )
            all_pg_ranks = _get_pg_ranks(
                [("tensor", tensor_size), ("pipe", pipe_size), ("data", data_size)],
                list(range(total_devices)),
                offset=0,
                total_size=total_devices,
            )

            tp_ranks = all_pg_ranks["tensor"][0]
            pipe_ranks = all_pg_ranks["pipe"][0]

            tensor_topo = self.device_topo.get_physical_topology(tp_ranks)
            pipe_topo = self.device_topo.get_physical_topology(pipe_ranks)
            (
                estimated_node_memory_reqs,
                estimated_node_flops_cost,
                estimated_intra_node_comm_cost,
                estimated_edge_comm_cost,
                contracted_graph,
            ) = self.profiler.fake_profile(
                mc.model,
                graph,
                sharding_specs,
                tensor_shapes,
                tensor_topo,
                self.fp32_flops,
                optimizer=optimizer,
            )
            # Do some scaling on the results
            # TODO The edge cost is dependent on the partition over the pipe_topo
            # We will use the average bandwidth on pipe_topo to do the scaling
            edge_cost_scale_factor = pipe_topo.get_average_bandwidth() / self.device_topo.intra_node_bandwidth
            estimated_edge_comm_cost *= edge_cost_scale_factor

            # Scale the memory, assuming num_stages / 2 chunks
            estimated_node_memory_reqs *= pipe_size / 2

            # flops cost does not have to be scaled

            # Run the unified TeraPipe partitioning for current tensor and pipe sizes
            partitions = tera_pipe_partition(
                len(estimated_node_memory_reqs),
                estimated_node_memory_reqs,
                estimated_node_flops_cost,
                estimated_intra_node_comm_cost,
                estimated_edge_comm_cost,
                pipe_size,
                self.memory_bound,
                gap_value=gap_value,
            )

            # If a valid partition is found, update optimal values
            if partitions:
                optimal_tensor_size = tensor_size
                optimal_pipe_size = pipe_size
                optimal_data_size = total_devices // current_product
                optimal_partitions = partitions
                optimal_product = current_product

        if pipe_size != 1 and partitions:
            nodes = list(contracted_graph.nodes())
            insert_before_nodes = [nodes[stage_range[-1] + 1] for stage_range in optimal_partitions[:-1]]
        else:
            insert_before_nodes = None
        return optimal_tensor_size, optimal_pipe_size, optimal_data_size, insert_before_nodes
