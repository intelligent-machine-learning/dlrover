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
from dlrover.python.common.log import default_logger as logger


class MemoryStore:
    """
    actor_names = []
    """

    def __init__(self, state_manager, jobname, namespace):
        self.__data_map = {}

    def get(self, key, default_value=None):
        return self.__data_map.get(key, default_value)

    def put(self, key, value):
        logger.info("putting key value {} and {}".format(key, value))
        self.__data_map[key] = value

    def delete(self, key):
        del self.__data_map[key]

    def add_actor_name(self, actor_type, actor_id, actor_name):
        actor_names = self.get("actor_names", {})
        actor_id_name_map = actor_names.get(actor_type, {})
        logger.info(
            "adding actor name to backend store"
            "actor_type {} actor_id {} and actor_name {}".format(
                actor_type, actor_id, actor_name
            )
        )
        actor_id_name_map.update({actor_id: actor_name})
        actor_names[actor_type] = actor_id_name_map
        self.put("actor_names", actor_names)
        logger.info(actor_names)
        return True

    def remove_actor_name(self, actor_name):
        actor_names = self.get("actor_names", {})
        for actor_type, name_list in actor_names.items():
            if actor_name in name_list:
                name_list.remove(actor_name)
                break
        self.put("actor_names", actor_names)
        logger.info("removing actor name %s from backend store" % actor_name)
        return True

    def do_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass


# actor_names = ["PsActor_0" ,"Worker-0|4"]
# RayClient
"""
负责查询Actor的状态
申请资源规格
增加删除Actor
"""
