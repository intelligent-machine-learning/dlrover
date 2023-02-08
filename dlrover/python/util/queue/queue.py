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

import queue
import threading

from dlrover.python.common.log import default_logger as logger


class ConcurrentQueue:
    def __init__(self, capacity=-1):
        self.__capacity = capacity
        self.__mutex = threading.Lock()
        self.__cond = threading.Condition(self.__mutex)
        self.__queue = queue.Queue()

    def get(self):
        if self.__cond.acquire():
            while self.__queue.empty():
                self.__cond.wait()
            elem = self.__queue.get()
            self.__cond.notify()
            self.__cond.release()
        return elem

    def put(self, elem):
        if self.__cond.acquire():
            while self.__queue.qsize() >= self.__capacity:
                self.__cond.wait()
            self.__queue.put(elem)
            self.__cond.notify()
            self.__cond.release()

    def clear(self):
        if self.__cond.acquire():
            self.__queue.queue.clear()
            self.__cond.release()
            self.__cond.notifyAll()

    def empty(self):
        is_empty = False
        if self.__mutex.acquire():
            is_empty = self.__queue.empty()
            self.__mutex.release()
        return is_empty

    def size(self):
        size = 0
        if self.__mutex.acquire():
            size = self.__queue.qsize()
            self.__mutex.release()
        return size

    def resize(self, capacity=-1):
        self.__capacity = capacity


class RayEventQueue:
    _instance_lock = threading.Lock()

    def __init__(self):
        self.queue = ConcurrentQueue(capacity=1000)

    def put(self, value):
        logger.info("putting {} into ray event queue".format(value))
        return self.queue.put(value)

    def get(self):
        value = self.queue.get()
        logger.info("getting {} into ray event queue".format(value))
        return value

    def size(self):
        return self.queue.size()

    @classmethod
    def singleton_instance(cls, *args, **kwargs):
        if not hasattr(RayEventQueue, "_instance"):
            with RayEventQueue._instance_lock:
                if not hasattr(RayEventQueue, "_instance"):
                    RayEventQueue._instance = RayEventQueue(*args, **kwargs)
        return RayEventQueue._instance
