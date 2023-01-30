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

import copy
import json
import threading
import time
from typing import Dict, List

 

from dlrover.python.common.constants import (
    DistributionStrategy,
    ElasticJobLabel,
    NodeEnv,
    NodeStatus,
    NodeType,
)
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.node import Node, NodeResource
from dlrover.python.master.scaler.base_scaler import ScalePlan, Scaler
from dlrover.python.scheduler.ray import RayClient

 
 

# 当前actor的信息存储在后端，通过后端查询当前有多少actor处于运行中
# 增加或者删除actor，根据index，从index最大到最小删除
# 

def parse_type(name):
    name = name.lower()
    node_type = None 
    if NodeType.PS in name:
        node_type = NodeType.PS
    elif NodeType.EVALUATOR in name :
        node_type = NodeType.EVALUATOR
    elif NodeType.WORKER in name :
        node_type = NodeType.WORKER
    return node_type

def parse_index(name):
    """
        PsActor_1 split("_")[-1]
        13-PythonOperator_streaming.operator.impl.tf_function.TFSinkFunction-4|20 split("|").split("-")[-1]
    """
    node_type = parse_type(name)
    node_index = None 
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
                # 通过 查询到当前正在运行的pod 
                if group_resource.count > len(cur_actors):
                    self._scale_up_actors(type, plan, cur_actors)
                elif group_resource.count < len(cur_actors):
                    self._scale_down_actor(type, plan, cur_actors)
 
 
    def _stats_alive_actors(self):
        job_pods: Dict[str, List[Node]] = {}
        resource = NodeResource(1, 1024) # to do 使用存储后端
        for name, status in self._ray_client.list_actor():
            actor_type, actor_index = parse_type_id_from_actor_name(name)
            node = Node(
                node_type=actor_type,
                node_id=name,
                name=name,
                rank_index=actor_index,
                status=status,
                start_time=None, #to be decided，获取actor创建时间
                config_resource=resource, #to be decided，获取actor的创建时间
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
        cur_num = len(cur_actors)
        
         
 

    def _scale_down_actors(
        self,
        type,
        plan: ScalePlan,
        cur_actors: List[Node],
    ):
        cur_num = len(cur_actors)
     
     