import json
import os
import threading
from queue import Queue

import redis  # type: ignore


class RMQ(object):
    def __init__(self, topic_name, ip="127.0.0.1", port=6379, queue=None):

        # pool = ConnectionPool.from_url(url=url, decode_responses=True)
        self.client = redis.Redis(
            host=ip, port=port, decode_responses=True  # <-- this will ensure that binary data is decoded
        )
        self.topic_name = topic_name
        self.queue = queue

    def publish(self, data, topic_name):
        self.client.publish(topic_name, data)

    def subscribe(self):
        pub = self.client.pubsub()
        pub.subscribe(self.topic_name)
        return pub

    def run_subscribe(self):
        pub = self.subscribe()
        for message in pub.listen():
            message_type = message["type"]
            if message_type == "subscribe":
                continue
            else:
                if message["data"] == "STOP":
                    print("Stopping subscriber...")
                    pub.unsubscribe()
                    break
                else:
                    self.queue.put(message["data"])


class RedisDataLoader:
    def __init__(self, ip="127.0.0.1", port=6379, batch_size=4):
        self.rank = os.environ.get("RANK", "0")
        self.queue = Queue()
        self.batch_size = batch_size
        self.redis_client = RMQ(self.rank, ip=ip, port=port, queue=self.queue)
        self.start_redis_client()

    def start_redis_client(self):
        t1 = threading.Thread(target=self.redis_client.run_subscribe, args=())
        t1.start()

    def __iter__(self):
        return self

    def __len__(self):
        # TODO get length from csv file
        return 100

    def __next__(self):
        batch = {"prompt": [], "prompt_att_mask": [], "output": [], "sample": []}
        while len(batch["prompt"]) < self.batch_size:
            ele = self.queue.get()
            ele = json.loads(ele)
            batch["prompt"].append(ele["prompt"])
            batch["output"].append(ele["output"])
            batch["sample"].append(ele["prompt"] + ele["output"])
        return batch
