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
import time
from typing import Dict, List

from kubernetes import client, config
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector

from dlrover.python.common.constants import NodeType
from dlrover.python.common.log_utils import default_logger as logger
from dlrover.python.common.node import NodeResource
from dlrover.python.common.singleton_utils import singleton
from dlrover.python.scheduler.job import ElasticJob

_SERVICE_PORTS = {
    NodeType.WORKER: 3333,
    NodeType.EVALUATOR: 3333,
    NodeType.CHIEF: 3333,
    NodeType.PS: 2222,
}

JOB_SUFFIX = "-edljob-"
ELASTICDL_APP_NAME = "elastic"
ELASTICDL_JOB_KEY = "elastic-job-name"
ELASTICDL_REPLICA_TYPE_KEY = "elastic-replica-type"
ELASTICDL_REPLICA_INDEX_KEY = "elastic-replica-index"


def get_pod_name(job_name, pod_type, worker_id):
    return "%s-%s" % (job_name + JOB_SUFFIX + pod_type, str(worker_id))


def append_pod_ip_to_env(env):
    pod_ip_var = V1EnvVar(
        name="MY_POD_IP",
        value_from=V1EnvVarSource(
            field_ref=V1ObjectFieldSelector(field_path="status.podIP")
        ),
    )
    node_ip_var = V1EnvVar(
        name="MY_NODE_IP",
        value_from=V1EnvVarSource(
            field_ref=V1ObjectFieldSelector(field_path="status.hostIP")
        ),
    )
    if env:
        env.append(pod_ip_var)
        env.append(node_ip_var)
    else:
        env = [pod_ip_var, node_ip_var]
    return env


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
            return self.client.create_namespaced_pod(self._namespace, pod)
        except client.rest.ApiException as e:
            logger.warning(
                "Failed to create %s pod: %s\n", pod.metadata.name, e
            )
            return None

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


class PodTemplate(object):
    """PodTemplate is the template of replica in a job"""

    def __init__(self, template):
        self.restart_policy = template["spec"]["restartPolicy"]
        main_container = template["spec"]["containers"][0]
        self.name = main_container["name"]
        self.image = main_container["image"]
        self.command = main_container["command"]
        self.image_pull_policy = main_container.get(
            "imagePullPolicy", "Always"
        )


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
        job = self._retry_to_get_job()
        self._distribution_strategy = job["spec"]["distributionStrategy"]
        self._replica_template: Dict[str, PodTemplate] = {}
        for replica, spec in job["spec"]["replicaSpecs"].items():
            self._replica_template[replica] = PodTemplate(spec["template"])

    def _get_common_labels(self):
        """Labels that should be attached to all k8s objects belong to
        current job.
        """
        return {"app": ELASTICDL_APP_NAME, ELASTICDL_JOB_KEY: self._job_name}

    def _retry_to_get_job(self):
        for _ in range(3):
            job = self._k8s_client.get_training_job()
            if job:
                return job
            else:
                time.sleep(5)
        raise ValueError("Cannot get the training job %s", self._job_name)

    def get_node_name(self, type, id):
        return get_pod_name(self._job_name, type, id)

    def get_service_name(self, pod_type, id):
        return self.get_node_name(pod_type, id)

    def get_node_service_addr(self, type, id):
        service_name = self.get_service_name(type, id)
        return "%s.%s.svc:%d" % (
            service_name,
            self._namespace,
            _SERVICE_PORTS[type],
        )

    def get_master_pod(self):
        master_pod_name = "elastic-%s-master" % self._job_name
        return self._k8s_client.get_pod(master_pod_name)

    def get_typed_pod(self, pod_type, id):
        pod_name = self.get_node_name(pod_type, id)
        return self._k8s_client.get_pod(pod_name)

    def create_typed_pod(self, pod_type, pod_id, resource: NodeResource):
        # Find that master pod that will be used as the owner reference
        # for the ps or worker pod.
        pod_name = self.get_node_name(pod_type, pod_id)
        master_pod = self.get_master_pod()
        env: List[V1EnvVar] = []
        env = append_pod_ip_to_env(env)

        lifecycle = None

        if pod_type not in self._replica_template:
            raise ValueError(
                "No replica %s specification in job %s",
                pod_type,
                self._job_name,
            )
        pod_template = self._replica_template[pod_type]

        labels = self._get_common_labels()

        pod = self.create_pod_obj(
            name=pod_name,
            image=resource.image,
            command=pod_template.command,
            resource_requests=resource.to_resource_dict(),
            resource_limits=resource.to_resource_dict(),
            priority=resource.priority,
            image_pull_policy=pod_template.image_pull_policy,
            restart_policy=pod_template.restart_policy,
            owner=master_pod,
            env=env,
            lifecycle=lifecycle,
            labels=labels,
        )
        # Add replica type and index
        pod.metadata.labels[ELASTICDL_REPLICA_TYPE_KEY] = pod_type
        pod.metadata.labels[ELASTICDL_REPLICA_INDEX_KEY] = str(pod_id)
        self._k8s_client.create_pod(pod)
        return pod

    def delete_typed_pod(self, pod_type, id):
        pod_name = self.get_node_name(pod_type, id)
        self._k8s_client.delete_pod(pod_name)

    def create_service(self, pod_type, pod_id, service_name):
        # Use master pod as service owner so the service will not be
        #  deleted if the corresponding pod is deleted.
        service = self._create_service_obj(
            name=service_name,
            port=_SERVICE_PORTS[pod_type],
            target_port=_SERVICE_PORTS[pod_type],
            replica_type=pod_type,
            replica_index=pod_id,
            owner=self.get_master_pod(),
        )
        self._k8s_client.create_service(service)
        return service

    def create_service_with_retry(
        self, pod_type, pod_id, service_name, retry_num=5
    ):
        for _ in range(retry_num):
            succeed = self.create_service(pod_type, pod_id, service_name)
            if succeed:
                return succeed
            else:
                time.sleep(5)
        return False

    def patch_service(self, pod_type, id, service_name):
        service = self._create_service_obj(
            name=service_name,
            port=_SERVICE_PORTS[pod_type],
            target_port=_SERVICE_PORTS[pod_type],
            replica_type=pod_type,
            replica_index=id,
            owner=self.get_master_pod(),
        )
        self._k8s_client.patch_service(service_name, service)

    def patch_service_with_retry(
        self, pod_type, new_id, service_name, retry_num=5
    ):
        for _ in range(retry_num):
            succeed = self.patch_service(pod_type, new_id, service_name)
            if succeed:
                return succeed
            else:
                time.sleep(5)
        return False

    def _create_service_obj(
        self,
        name,
        port,
        target_port,
        replica_type,
        replica_index,
        owner=None,
    ):
        labels = self._get_common_labels()

        metadata = client.V1ObjectMeta(
            name=name,
            labels=labels,
            # Note: We have to add at least one annotation here.
            # Otherwise annotation is `None` and cannot be modified
            # using `with_service()` for cluster specific information.
            annotations=labels,
            owner_references=self.create_owner_reference(owner)
            if owner
            else None,
            namespace=self._namespace,
        )
        selector = {
            "app": ELASTICDL_APP_NAME,
            ELASTICDL_JOB_KEY: self._job_name,
            ELASTICDL_REPLICA_TYPE_KEY: replica_type,
            ELASTICDL_REPLICA_INDEX_KEY: str(replica_index),
        }
        spec = client.V1ServiceSpec(
            ports=[client.V1ServicePort(port=port, target_port=target_port)],
            selector=selector,
            type=None,
        )
        service = client.V1Service(
            api_version="v1", kind="Service", metadata=metadata, spec=spec
        )
        return service

    def create_persistent_volume_claim(
        self, pvc_name, storage, iolimits_enabled
    ):
        # Use master pod as service owner so the service will not be
        #  deleted if the corresponding pod is deleted.
        pvc = self._create_persistent_volume_claim_object(
            name=pvc_name,
            storage=storage,
            owner=self.get_master_pod(),
            iolimits_enabled=iolimits_enabled,
        )
        return self._k8s_client.create_pvc(pvc)

    def _create_persistent_volume_claim_object(
        self, name, storage, owner, iolimits_enabled
    ):
        labels = self._get_common_labels()
        # Support speed limitation of SSD
        annotations = {}
        if iolimits_enabled:
            annotations["alibabacloud.csi.alipay.com/enable-iolimits"] = "true"
        metadata = client.V1ObjectMeta(
            name=name,
            labels=labels,
            # Note: We have to add at least one annotation here.
            # Otherwise annotation is `None` and cannot be modified
            # using `with_service()` for cluster specific information.
            annotations=annotations,
            owner_references=self.create_owner_reference(owner)
            if owner
            else None,
            namespace=self._namespace,
        )

        spec = client.V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=client.V1ResourceRequirements(
                requests={"storage": storage}
            ),
            storage_class_name="alibabacloud-raw-file",
            volume_mode="Filesystem",
        )
        pvc = client.V1PersistentVolumeClaim(
            api_version="v1",
            kind="PersistentVolumeClaim",
            metadata=metadata,
            spec=spec,
        )
        return pvc

    def get_job_uuid(self):
        job_data = self._k8s_client.get_training_job()
        if (
            job_data
            and "metadata" in job_data
            and "uid" in job_data["metadata"]
        ):
            return job_data["metadata"]["uid"]
        return ""

    def create_pod_obj(
        self,
        name,
        owner,
        image,
        command,
        resource_requests: Dict[str, float],
        resource_limits: Dict[str, float],
        image_pull_policy,
        lifecycle,
        env,
        restart_policy,
        priority,
        labels,
        termination_period=None,
    ):
        resource_limits = (
            resource_limits if len(resource_limits) > 0 else resource_requests
        )
        container = client.V1Container(
            name="main",
            image=image,
            command=command,
            resources=client.V1ResourceRequirements(
                requests=resource_requests,
                limits=resource_limits,
            ),
            image_pull_policy=image_pull_policy,
            env=env,
            lifecycle=lifecycle,
        )

        # Pod
        spec = client.V1PodSpec(
            containers=[container],
            restart_policy=restart_policy,
            priority_class_name=priority,
            termination_grace_period_seconds=termination_period,
        )

        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            spec=spec,
            metadata=client.V1ObjectMeta(
                name=name,
                labels=labels,
                owner_references=self.create_owner_reference(owner),
                namespace=self._namespace,
            ),
        )
        return pod

    @staticmethod
    def create_owner_reference(owner_pod):
        owner_ref = (
            [
                client.V1OwnerReference(
                    api_version="v1",
                    block_owner_deletion=True,
                    kind="Pod",
                    name=owner_pod.metadata.name,
                    uid=owner_pod.metadata.uid,
                )
            ]
            if owner_pod
            else None
        )
        return owner_ref
