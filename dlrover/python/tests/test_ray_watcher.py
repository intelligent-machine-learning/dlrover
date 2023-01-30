# Copyright 2022 The EasyDL Authors. All rights reserved.
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
import unittest
from typing import List
 

 
from dlrover.python.common.node import Node
 
from dlrover.python.master.watcher.ray_watcher import (
    ActorWatcher
)

from dlrover.python.scheduler.ray import RayClient
from dlrover.python.util.queue.queue import RayEventQueue

class ActorWatcherTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.system("ray stop --force")
        r = os.system("ray start --head --port=5001  --dashboard-port=5000")
        cls.actor_watcher = ActorWatcher("test", "")

    @classmethod
    def tearDownClass(cls):
        os.system("ray stop --force")

    def test_list(self):
       
        nodes: List[Node] = self.actor_watcher.list()
        self.assertEqual(len(nodes), 0)
        self.ray_client = RayClient.singleton_instance()

        class A:
            def __init__(self):
                self.a = "a"
            
            def run(self):
                return self.a
            
            def health_check(self,*args,**kargs):
                return "a"
        # to do: 使用类来描述启动的参数而不是dict
        actor_args = {"executor":A, "actor_name":"worker-0|1"}
        self.ray_client.create_actor(actor_args=actor_args)
        import time
        time.sleep(1)
        nodes: List[Node] = self.actor_watcher.list()
        self.assertEqual(len(nodes), 1)

    def test_watch(self):
        ray_event_queue = RayEventQueue.singleton_instance()
        events = ["1","2","3","4"]
        for e in events:
            ray_event_queue.put(e)
            
        self.assertEqual(self.actor_watcher.event_queue.size(),len(events))
        


        
 

if __name__ == "__main__":
    unittest.main()