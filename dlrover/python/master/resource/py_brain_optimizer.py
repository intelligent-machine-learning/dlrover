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

from distutils.log import log
from logging import exception
import math
import uuid
import os
import time

import requests
from dlrover.brain.python.client.client import BrainClient

from dlrover.python.common.constants import MemoryUnit, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import NodeGroupResource, NodeResource
from dlrover.python.master.resource.optimizer import (
    ResourceOptimizer,
    ResourcePlan,
)

from dlrover.brain.python.common.http_schemas import (
    OptimizeRequest,
    OptimizeResponse,
    Response,
)
from dlrover.brain.python.common.job import (
    JobMeta,
    JobOptimizePlan,
)
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def catch_brain_optimization_exception(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.warning("Fail to execute %s by %s", func.__name__, e)
            return ResourcePlan()

    return wrapper


def convert_plan_msg(plan_msg):
    """

    """
    plan = ResourcePlan()
    if not plan_msg:
        return plan
    for type, group in plan_msg.job_resource.node_group_resources.items():
        if type != NodeType.WORKER:
            continue
        count = group.count
        memory = int(group.resource.memory/MemoryUnit.MB)  # MiB
        cpu = math.ceil(group.resource.cpu)
        gpu_num = int(group.resource.gpu)
        gpu_type = group.resource.gpu_type
        priority = group.resource.priority
        logger.info(f"type: {type}, count: {count}, memory: {memory}Mi, cpu: {cpu}, gpu: {gpu_num}, gpu_type: {gpu_type}, priority: {priority}")
        plan.node_group_resources[type] = NodeGroupResource(
            count, NodeResource(cpu, memory, gpu_type, gpu_num, priority)
        )
    return plan


class PyBrainResoureOptimizer(ResourceOptimizer):
    """Query resource plan from the brain service.
       request:
        {
            "type": "standard",
            "job": {
                "uuid": "job-uuid-123",
                "cluster": "prod-cluster",
                "namespace": "default"
            },
            "config": {
                "optimizer": "BaseOptimizer",
                "customized_config": {"pop_size": "50"}
            }
        }
    """

    def __init__(self, job_uuid, job_name, cluster, namespace, resource_limits):
        super(PyBrainResoureOptimizer, self).__init__(job_uuid, resource_limits)
        self.job_name = job_name
        self.cluster = cluster
        self.namespace = namespace
        self.brain_server_address = self.get_brain_server_address()
        self._brain_client = BrainClient(self.brain_server_address)

    def get_brain_server_address(self):
        """
            Get the brain server address from the environment variable.
            example: "http://dlrover-brain.dlrover.svc.cluster.local:50002"
        """
        service_name = os.getenv("DLROVER_BRAIN_SERVICE_NAME", "dlrover-brain")
        brain_service_port = int(os.getenv("DLROVER_BRAIN_SERVICE_PORT", "50002"))
        return "http://%s.%s.svc.cluster.local:%d" % (
            service_name,
            self.namespace,
            brain_service_port,
        )

    @catch_brain_optimization_exception
    def generate_opt_plan(self, stage, config={}) -> ResourcePlan:
        pass

    @catch_brain_optimization_exception
    def generate_oom_recovery_plan(self, oom_nodes, stage, config={}):
       pass
    
    @catch_brain_optimization_exception
    def generate_resource_plan_with_optimizer(self, config):
        job = JobMeta(uuid=self._job_uuid, cluster=self.cluster, namespace=self.namespace)
        req_data = OptimizeRequest(job=job, config=config)

        logger.info(f"Request brain server with url {self.brain_server_address}")
        # interface timeout
        timeout_seconds = int(os.getenv("BRAIN_INTERFACE_TIMEOUT_SECONDS", "60"))

        # asyn request brain optimize
        with ThreadPoolExecutor(max_workers=3) as executor:
            future = executor.submit(self._brain_client.optimize, req_data)
            try:
                res: OptimizeResponse = future.result(timeout=timeout_seconds)
            except TimeoutError:
                logger.error("Request to brain server timed out after %d seconds", timeout_seconds)
                return ResourcePlan()

        if not res or not res.job_opt_plan:
            logger.warning("No any optimization plan")
            return ResourcePlan()

        plan_msg = res.job_opt_plan
        logger.info(
            "The optimization plan with config %s is %s",
            config,
            plan_msg,
        )
        plan = convert_plan_msg(plan_msg)
        return plan
