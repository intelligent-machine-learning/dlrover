import networkx as nx


class DeviceTopology:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_device(self, device_id, attributes):
        self.graph.add_node(device_id, **attributes)

    def connect_devices(self, device_id_1, device_id_2, bandwidth):
        self.graph.add_edge(device_id_1, device_id_2, bandwidth=bandwidth, bandwidth_inverse=1.0 / bandwidth)

    def get_physical_topology(self, device_ids):
        subgraph = self.graph.subgraph(device_ids)
        sub_topology = DeviceTopology()
        sub_topology.graph = subgraph
        return sub_topology

    def num_devices(self):
        return self.graph.number_of_nodes()

    def get_device_ranks(self):
        return list(self.graph.nodes)

    def get_effective_bandwidth(self, device_id_1: int, device_id_2: int):
        # Use the inverse bandwidth to compute the shortest path
        # Use the bottleneck on the path as the effective bandwidth
        if device_id_1 == device_id_2:
            return float("inf")

        path = nx.shortest_path(self.graph, device_id_1, device_id_2, weight="bandwidth_inverse")
        return min(self.graph.edges[path[i], path[i + 1]]["bandwidth"] for i in range(len(path) - 1))

    def get_average_bandwidth(self):
        average_bandwidth = 0.0
        counter = 0
        ranks = self.get_device_ranks()
        for i in self.get_device_ranks():
            for j in range(i + 1, self.num_devices()):
                bandwidth = self.get_effective_bandwidth(ranks[i], ranks[j])
                if bandwidth != float("inf"):
                    counter += 1
                    average_bandwidth += (bandwidth - average_bandwidth) / counter
        return average_bandwidth


class SimpleTopology(DeviceTopology):
    def __init__(self, num_nodes, num_devices_per_node, intra_node_bandwidth, inter_node_bandwidth):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_devices_per_node = num_devices_per_node
        self.intra_node_bandwidth = intra_node_bandwidth
        self.inter_node_bandwidth = inter_node_bandwidth
        self.build_network()

    def build_network(self):
        total_devices = self.num_nodes * self.num_devices_per_node
        device_ids = list(range(total_devices))

        # Add all devices to the network
        for device_id in device_ids:
            node_id = device_id // self.num_devices_per_node
            self.add_device(device_id, {"node_id": node_id})

        # Connect all devices within the same node with high bandwidth
        for node_id in range(self.num_nodes):
            node_devices = [
                device_id
                for device_id in range(node_id * self.num_devices_per_node, (node_id + 1) * self.num_devices_per_node)
            ]
            for i in range(len(node_devices)):
                for j in range(i + 1, len(node_devices)):
                    self.connect_devices(node_devices[i], node_devices[j], self.intra_node_bandwidth)
                    self.connect_devices(node_devices[j], node_devices[i], self.intra_node_bandwidth)

        # Connect all devices in different nodes with low bandwidth
        for i in range(total_devices):
            for j in range(i + 1, total_devices):
                if self.graph.nodes[i]["node_id"] != self.graph.nodes[j]["node_id"]:
                    self.connect_devices(i, j, self.inter_node_bandwidth)
                    self.connect_devices(j, i, self.inter_node_bandwidth)
