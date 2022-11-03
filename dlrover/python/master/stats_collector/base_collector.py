from abc import ABCMeta, abstractmethod


class StatsCollector(metaclass=ABCMeta):
    def __init__(self, job_uuid):
        self._job_uuid = job_uuid
        self._node_resource_usage = {}

    def collect_node_resource_usage(self, node_name, resource):
        """Collect the resource usage of the node_name node.
        Args:
            node_name: string, the name of node
            resource: elasticl.python.resource.NodeResource instace,
                the resource usage of the node.
        """
        self._node_resource_usage[node_name] = resource

    @abstractmethod
    def report_resource_usage(self):
        pass
