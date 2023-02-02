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

import logging
import sys
import time

import ray

_logger: logging.Logger = logging.getLogger(__name__)
_logger.setLevel("INFO")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# 可以用来查询查询actor


# 具体的worker，RayWorker
@ray.remote
class Run:
    def __init__(self, name):
        self.name = name
        # 具体的执行可以是estimatorexecutor

    def run(self):
        logging.info("abcdefgh")
        time.sleep(2)
        return "run"


# 相当于master的角色  /是一个actor，可以通过提交作业的方式拉起，也可以通过remote的方法创建
class RayWorker:
    def __init__(self):
        self.actor = 1
        self.actor_names = []

    def schedule(self):

        for i in range(3):
            actor_name = "actor{}".format(i)
            self.actor_names.append(actor_name)
            r = Run.options(name=actor_name).remote(i)
            actor_id = ray.get_actor(actor_name)
            logging.error("actor_id  {}".format(actor_id))
            o = r.run.remote()
            logging.error("getting o {}".format(ray.get(o)))

            actor_id = ray.get_actor("not_exist")
            logging.error("actor_id  {}".format(actor_id))


if __name__ == "__main__":
    ray_worker = RayWorker()
    logging.error("aaaaaaaaaaa")
    ray_worker.schedule()
