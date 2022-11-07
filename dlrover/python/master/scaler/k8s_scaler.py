from dlrover.python.master.scaler.base_scaler import Scaler
from abc import ABCMeta, abstractmethod


class BaseScalerSpec(metaclass=ABCMeta):
    @abstractmethod
    def to_dict(self):
        """Serialize the object to a dictionary"""
        pass


class ScalerResourceSpec(BaseScalerSpec):
    def __init__(self, cpu, memory, gpu):
        self.cpu = cpu
        self.memory = memory
        self.gpu = gpu

    def to_dict(self):
        return self.__dict__()


class ScalerReplicaResourceSpec(BaseScalerSpec):
    def __init__(self, replicas, resource_spec) -> None:
        self.replicas = replicas
        self.resource = resource_spec

    def to_dict(self):
        spec = {}
        spec["replicas"] = self.replicas
        spec["resource"] = self.resource.to_dict()
        return spec


class ScalerSpec(BaseScalerSpec):
    def __init__(self, owner_job) -> None:
        self.owner_job = owner_job
        self.replica_resource_specs = {}
        self.node_resource_specs = {}

    def add_replica_resource(self, replica_name, replica_resource_spec):
        self.replica_resource_specs[replica_name] = replica_resource_spec

    def add_node_resource(self, node_name, resource_spec):
        self.node_resource_specs[node_name] = resource_spec

    def to_dict(self):
        spec = {}
        spec["ownerJob"] = self.owner_job
        spec["replicaResourceSpec"] = {}
        for name, resource_spec in self.replica_resource_specs.items():
            spec["replicaResourceSpec"][name] = resource_spec.to_dict()
        spec["nodeResourceSpec"] = {}
        for name, resource_spec in self.node_resource_specs.items():
            spec["nodeResourceSpec"] = resource_spec.to_dict()
        return spec


class ScalerKind(object):
    """ScalerKind is a dictionary of a Scaler CRD for an ElasticJob
    to scale up/down Pods of a job on a k8s cluster.
    """
    def __init__(self, api_version=None, kind=None, metadata=None, spec=None):
        self._api_version = api_version
        self._kind = kind
        self._metadata = metadata
        self._spec = spec


class k8sScaler(Scaler):
    def __init__(self, job_name, namespace, cluster):
        self._namespace = namespace
        self._cluster = cluster
        super(k8sScaler, self).__int__(job_name)

    def scale(self, resource_plan):
        pass