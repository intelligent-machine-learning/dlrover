# Copyright 2023 The DLRover Authors. All rights reserved.
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

import threading
from typing import Dict, List

import ray

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.scheduler.ray import RayClient

#  通过反射获取，而不是import包
from dlrover.python.util.reflect_util import get_class


class ActorArgs:
    def __init__(self, actor_name, executor, args=[], kargs={}):
        self.actor_name = actor_name
        self.executor = executor
        self.args = args
        self.kargs = kargs

    def get(self, key, default_value=None):
        return getattr(self, key, default_value)


def parse_type(name) -> str:
    name = name.lower()
    node_type: str = ""
    if NodeType.PS in name:
        node_type = NodeType.PS
    elif NodeType.EVALUATOR in name:
        node_type = NodeType.EVALUATOR
    elif NodeType.WORKER in name:
        node_type = NodeType.WORKER
    return node_type


def parse_index(name) -> int:
    """
    PsActor_1 split("_")[-1]
    TFSinkFunction-4|20 split("|").split("-")[-1]
    """
    node_type = parse_type(name)
    node_index: int = 0
    if node_type == NodeType.PS:
        node_index = int(name.split("_")[-1])
    elif node_type == NodeType.EVALUATOR:
        node_index = 1
    elif node_type == NodeType.WORKER:
        node_index = int(name.split("|")[0].split("-")[-1])
    return node_index


def parse_type_id_from_actor_name(name):
    node_type = parse_type(name)
    node_index = parse_index(name)
    return node_type, node_index


ray.init()


class ActorScaler(Scaler):
    """ActorScaler launches or removes actor using Ray APIs."""

    def __init__(self, job_name, namespace):
        super(ActorScaler, self).__init__(job_name)
        self._ray_client = RayClient.singleton_instance(namespace, job_name)
        self._namespace = namespace
        self._lock = threading.Lock()

    def _retry_to_get_job(self):
        pass

    def scale(self, plan: ScalePlan):
        self._plan = plan
        logger.info("Scale the job by plan %s", plan.toJSON())
        with self._lock:
            alive_actors = self._stats_alive_actors()
            for type, group_resource in plan.node_group_resources.items():
                cur_actors = alive_actors.get(type)
                cur_actors = []
                if group_resource.count > len(alive_actors):
                    self._scale_up_actors(type, plan, cur_actors)
                elif group_resource.count < len(alive_actors):
                    self._scale_down_actors(type, plan, cur_actors)

    def _stats_alive_actors(self):
        job_pods: Dict[str, List[Node]] = {}
        resource = NodeResource(1, 1024)  # to do 使用存储后端
        for name, status in self._ray_client.list_actor():
            actor_type, actor_index = parse_type_id_from_actor_name(name)
            node = Node(
                node_type=actor_type,
                node_id=name,
                name=name,
                rank_index=actor_index,
                status=status,
                start_time=None,  # to be decided，获取actor创建时间
                config_resource=resource,  # to be decided，获取actor的创建时间
            )
            if node.status in [
                NodeStatus.PENDING,
                NodeStatus.RUNNING,
                NodeStatus.SUCCEEDED,
            ]:
                job_pods[actor_type].append(node)

        return job_pods

    def _scale_up_actors(
        self,
        type,
        plan: ScalePlan,
        cur_actors: List[Node],
    ):
        print(cur_actors)
        v = plan.node_group_resources[type]
        actor_name: str = ""
        if v.count > 0:
            context = {
                "platform": "ray",
                "ps_num": 1,
                "worker_num": 1,
                "conf": "conf.TrainConf",
                "task_id": 0,
                "task_type": type,
            }
            if type == "ps":
                actor_name = "PsActor_0"
            elif type == "chief":
                actor_name = "PythonWorker-0|1"
            TFRayWorker = get_class(
                "dlrover.trainer.worker.tf_ray_worker.TFRayWorker"
            )
            actor_args = ActorArgs(
                actor_name=actor_name,
                executor=TFRayWorker,
                args=[],
                kargs={"args": context},
            )
            self._ray_client.create_actor(actor_args=actor_args)

    def _scale_down_actors(
        self,
        type,
        plan: ScalePlan,
        cur_actors: List[Node],
    ):
        print(type)
        print(plan)
        print(cur_actors)
