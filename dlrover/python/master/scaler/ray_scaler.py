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
from typing import Dict, List, Optional

from dlrover.python.common.constants import NodeStatus, NodeType
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.scheduler.ray import RayClient
from dlrover.python.util.actor_util.parse_actor import (
    parse_type_id_from_actor_name,
)
from dlrover.python.util.reflect_util import get_class


class ActorArgs:
    def __init__(self, actor_name, executor, args=[], kargs={}):
        self.actor_name = actor_name
        self.executor = executor
        self.args = args
        self.kargs = kargs

    def get(self, key, default_value=None):
        return getattr(self, key, default_value)


class ActorScaler(Scaler):
    """ActorScaler launches or removes actor using Ray APIs."""

    def __init__(self, job_name, namespace):
        super(ActorScaler, self).__init__(job_name)
        self._ray_client = RayClient.singleton_instance(namespace, job_name)
        self._namespace = namespace
        self._lock = threading.Lock()

    def start(self):
        pass

    def scale(self, plan: ScalePlan, **kwargs):
        self._plan = plan
        logger.info("Scale the job by plan %s", plan.to_json())
        with self._lock:
            alive_actors = self._stats_alive_actors()
            for type, group_resource in plan.node_group_resources.items():
                cur_actors = alive_actors.get(type)
                if group_resource.count > len(alive_actors):
                    self._scale_up_actors(type, plan, cur_actors)
                elif group_resource.count < len(alive_actors):
                    logger.info(group_resource.count)
                    self._scale_down_actors(type, plan, cur_actors)

    def _stats_alive_actors(self) -> Dict[str, List[Node]]:
        job_pods: Dict[str, List[Node]] = {}

        for name, status in self._ray_client.list_actor():
            actor_type, actor_index = parse_type_id_from_actor_name(name)
            if actor_type not in job_pods:
                job_pods[actor_type] = []
            node = Node(
                node_type=actor_type,
                node_id=name,
                name=name,
                rank_index=actor_index,
                status=status,
                start_time=None,
            )
            if node.status in [
                NodeStatus.PENDING,
                NodeStatus.RUNNING,
                NodeStatus.SUCCEEDED,
            ]:
                job_pods[actor_type].append(node)
            job_pods[actor_type].append(node)
        return job_pods

    def _scale_up_actors(
        self,
        type,
        plan: ScalePlan,
        cur_actors: Optional[List[Node]],
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
            if type == NodeType.PS:
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
        cur_actors: Optional[List[Node]],
    ):
        logger.info(ScalePlan)
        if type == NodeType.PS:
            actor_name = "PsActor_0"
        elif type == NodeType.WORKER:
            actor_name = "PythonWorker-0|1"
        self._ray_client.delete_actor(actor_name)
