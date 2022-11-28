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

from abc import ABCMeta, abstractclassmethod


class ElasticJob(metaclass=ABCMeta):
    def __init__(self, namespace, job_name):
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
            force_use_kube_config_file: If true, force to load the cluster
                config from ~/.kube/config. Otherwise, if it's in a process
                running in a K8S environment, it loads the incluster config,
                if not, it loads the kube config file.
        """
        self._namespace = namespace
        self._job_name = job_name

    @abstractclassmethod
    def get_job_uuid(self):
        pass

    @abstractclassmethod
    def get_node_service_addr(self, type, id):
        pass

    @abstractclassmethod
    def get_node_name(self, type, id):
        pass
