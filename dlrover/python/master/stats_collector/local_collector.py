from dlrover.python.master.stats_collector.base_collector import StatsCollector


class LocalStatsCollector(StatsCollector):
    def __init__(self, job_uuid):
        super(LocalStatsCollector, self).__init__(job_uuid)
        self._all_node_resources = {}

    def report_resource_usage(self):
        for node_name, resource in self._node_resource_usage.items():
            self._all_node_resources.setdefault(node_name, [])
            self._all_node_resources[node_name].append(resource)
