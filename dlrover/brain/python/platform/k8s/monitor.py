from kubernetes import client, config
from kubernetes.client.rest import ApiException
from dlrover.brain.python.common.log import default_logger as logger


class ResourceMonitor(object):
    def __init__(self, namespace, gpu_resource_name):
        self.namespace = namespace
        self.gpu_resource_name = gpu_resource_name
        # load K8s config
        # In cluster，use config.load_incluster_config()
        # local，use config.load_kube_config()
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config() # local kubeconfig mode
            

        self.v1 = client.CoreV1Api()

    def get_cluster_idle_gpus(self) -> int:
        """
            get idle gpus in cluster
            algorithm：idles = Capacity - Allocated
        """
        try:
            nodes = self.v1.list_node()
            
            total_idle = 0
            for node in nodes.items:
                node_name = node.metadata.name

                # Get the total allocatable GPUs on the node (Allocatable), which is usually equal to the number of physical GPUs installed.
                max_gpus = 0
                if node.status.allocatable and self.gpu_resource_name in node.status.allocatable:
                    max_gpus = int(node.status.allocatable[self.gpu_resource_name])

                # Filter out the running pods scheduled to this node.
                field_selector = f"spec.nodeName={node_name}"
                pods = self.v1.list_pod_for_all_namespaces(field_selector=field_selector)
                
                # Get the number of currently allocated GPUs.
                current_allocated_gpus = 0
                for pod in pods.items:
                    if pod.status.phase not in ['Running', 'Pending']:
                        continue
                    
                    for container in pod.spec.containers:
                        if container.resources and container.resources.requests:
                            request_gpus = container.resources.requests.get(self.gpu_resource_name)
                            if request_gpus:
                                current_allocated_gpus += int(request_gpus)
                
                idle_gpu_count = max_gpus - current_allocated_gpus
                
                idle_count = max(0, idle_gpu_count)

                total_idle += idle_count

                logger.info(
                    f"total_gpuas:{max_gpus}," 
                    f"current_allocated_gpus:{current_allocated_gpus}," 
                    f"idle_gpus:{idle_count}"
                )

            logger.info(f"cluster total idle gpus:{total_idle}")
            
            return total_idle

        except ApiException as e:
            logger.error(f"Call k8s API exception：{e}")
            return 0
        except Exception as e:
            logger.error(f"unknown exception：{e}")
            return 0