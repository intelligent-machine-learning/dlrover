# Copyright 2022 The DLRover Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from kubernetes import client, config

from dlrover.python.common.constants import NodeType
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.common.singleton_utils import singleton
from dlrover.python.scheduler.job import ElasticJob

NODE_SERVICE_PORTS = {
    NodeType.WORKER: 3333,
    NodeType.EVALUATOR: 3333,
    NodeType.CHIEF: 3333,
    NodeType.PS: 2222,
}

JOB_SUFFIX = "-edljob-"


def get_pod_name(job_name, pod_type, worker_id):
    return "%s-%s" % (job_name + JOB_SUFFIX + pod_type, str(worker_id))


@singleton
class k8sClient(object):
    def __init__(
        self,
        namespace,
        job_name,
        force_use_kube_config_file=False,
    ):
        """
        ElasticDL k8s client.

        Args:
            image_name: Docker image path for ElasticDL pod.
            namespace: The name of the Kubernetes namespace where ElasticDL
                pods will be created.
            job_name: ElasticDL job name, should be unique in the namespace.
                Used as pod name prefix and value for "elastic" label.
            event_callback: If not None, an event watcher will be created and
                events passed to the callback.
            periodic_call_func: If not None, call this method periodically.
            force_use_kube_config_file: If true, force to load the cluster
                config from ~/.kube/config. Otherwise, if it's in a process
                running in a K8S environment, it loads the incluster config,
                if not, it loads the kube config file.
        """
        try:
            if (
                os.getenv("KUBERNETES_SERVICE_HOST")
                and not force_use_kube_config_file
            ):
                # We are running inside a k8s cluster
                config.load_incluster_config()
                logger.info("Load the incluster config.")
            else:
                # Use user's kube config
                config.load_kube_config()
                logger.info("Load the kube config file.")
        except Exception as ex:
            logger.error(
                "Failed to load configuration for Kubernetes:\n%s", ex
            )

        self.client = client.CoreV1Api()
        self.api_instance = client.CustomObjectsApi()
        self._namespace = namespace
        self._job_name = job_name

    def list_namespaced_pod(self, label_selector):
        try:
            pod_list = self.client.list_namespaced_pod(
                self._namespace,
                label_selector=label_selector,
            )
            return pod_list
        except Exception as e:
            logger.warning(e)
        return None

    def create_custom_resource(self, group, version, plural, body):
        try:
            print(body)
            self.api_instance.create_namespaced_custom_object(
                group,
                version,
                "default",
                plural,
                body,
            )
        except client.rest.ApiException as e:
            logger.error(
                "Exception when calling CustomObjectsApi->",
                "create_namespaced_custom_object: %s" % e,
            )

    def get_custom_resource(self, name, group, version, plural):
        try:
            crd_object = self.api_instance.get_namespaced_custom_object(
                namespace=self._namespace,
                name=name,
                group=group,
                version=version,
                plural=plural,
            )
            return crd_object
        except client.ApiException as e:
            logger.warning("Exception when getting custom object: %s\n" % e)
            return None

    def get_training_job(self):
        try:
            crd_object = self.get_custom_resource(
                name=self._job_name,
                group="jobs.kubemaker.alipay.net",
                version="v1beta1",
                plural="elasticjobs",
            )
            return crd_object
        except client.ApiException as e:
            logger.warning("Exception when getting custom object: %s\n" % e)
            return None

    def get_configmap(self, name):
        try:
            configmap = self.client.read_namespaced_config_map(
                namespace=self._namespace, name=name
            )
            return configmap
        except client.ApiException as e:
            logger.warning("Exception when getting configmap: %s\n" % e)
            return None

    def create_pod(self, pod):
        try:
            self.client.create_namespaced_pod(self._namespace, pod)
            return True
        except client.rest.ApiException as e:
            logger.warning(
                "Failed to create %s pod: %s\n", pod.metadata.name, e
            )
            return False

    def get_pod(self, name):
        try:
            return self.client.read_namespaced_pod(
                namespace=self._namespace, name=name
            )
        except client.ApiException as e:
            logger.warning("Exception when reading pod %s: %s\n" % (name, e))
            return None

    def delete_pod(self, name):
        try:
            self.client.delete_namespaced_pod(
                name,
                self._namespace,
                body=client.V1DeleteOptions(),
            )
            return True
        except Exception as e:
            logger.warning(e)
            return False

    def patch_labels_to_pod(self, name, labels_dict):
        body = {"metadata": {"labels": labels_dict}}
        try:
            return self.client.patch_namespaced_pod(
                name=name, namespace=self._namespace, body=body
            )
        except client.ApiException as e:
            logger.warning("Exception when patching labels to pod: %s\n" % e)
            return None

    def patch_annotations_to_pod(self, name, annotations):
        body = {"metadata": {"annotations": annotations}}
        try:
            return self.client.patch_namespaced_pod(
                name=name, namespace=self._namespace, body=body
            )
        except client.ApiException as e:
            logger.warning(
                "Exception when patching annotations to pod: %s\n" % e
            )
            return None

    def create_service(self, service):
        try:
            self.client.create_namespaced_service(self.namespace, service)
            return True
        except client.rest.ApiException as e:
            logger.warning(
                "Failed to create %s service: %s\n"
                % (service.metadata.name, e)
            )
            return False

    def patch_service(self, service_name, service):
        try:
            self.client.patch_namespaced_service(
                service_name, self.namespace, service
            )
            return True
        except client.rest.ApiException as e:
            logger.warning(
                "Failed to patch %s service: %s\n" % (service_name, e)
            )
            return False

    def get_service(self, name):
        try:
            return self.client.read_namespaced_service(
                # worker service has the same name as pod name
                name=name,
                namespace=self._namespace,
            )
        except client.ApiException:
            return None

    def create_pvc(self, pvc):
        try:
            self.client.create_namespaced_persistent_volume_claim(
                self._namespace, pvc
            )
            return True
        except client.rest.ApiException as e:
            logger.warning(
                "Failed to create %s persistent volume claim: %s\n"
                % (pvc.metadata.name, e)
            )
            return False


class K8sElasticJob(ElasticJob):
    def __init__(self, job_name, namespace):
        """
        ElasticJob manages Pods by K8s Python APIs. The example of an elastic
        job is in dlrover/go/elasticjob_operator/config/samples/
        elastic_v1alpha1_elasticjob.yaml
        Args:
            image_name: Docker image path for ElasticDL pod.
            namespace: The name of the Kubernetes namespace where ElasticDL
                pods will be created.
            job_name: ElasticDL job name, should be unique in the namespace.
                Used as pod name prefix and value for "elastic" label.
        """
        self._k8s_client = k8sClient(namespace, job_name)
        self._namespace = namespace
        self._job_name = job_name

    def get_node_name(self, type, id):
        return get_pod_name(self._job_name, type, id)

    def get_node_service_addr(self, type, id):
        service_name = get_pod_name(self._job_name, type, id)
        return "%s.%s.svc:%d" % (
            service_name,
            self._namespace,
            NODE_SERVICE_PORTS[type],
        )

    def get_job_uuid(self):
        job_data = self._k8s_client.get_training_job()
        if (
            job_data
            and "metadata" in job_data
            and "uid" in job_data["metadata"]
        ):
            return job_data["metadata"]["uid"]
        return ""
