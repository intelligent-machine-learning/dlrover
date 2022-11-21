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

JOB_SUFFIX = "-edljob-"


_SERVICE_PORTS = {
    NodeType.WORKER: 3333,
    NodeType.EVALUATOR: 3333,
    NodeType.TF_MASTER: 3333,
    NodeType.PS: 2222,
}


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
                Used as pod name prefix and value for "elasticdl" label.
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
                plural="trainings",
            )
            return crd_object
        except client.ApiException as e:
            logger.warning("Exception when getting custom object: %s\n" % e)
            return None

    def get_service_address(self, pod_type, pod_id):
        service_name = self.get_pod_name(pod_type, pod_id)
        return "%s.%s.svc:%d" % (
            service_name,
            self._namespace,
            _SERVICE_PORTS[pod_type],
        )

    def get_pod_name(self, pod_type, pod_id):
        return "%s-%s" % (self._job_name + JOB_SUFFIX + pod_type, pod_id)
