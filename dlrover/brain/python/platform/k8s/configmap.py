# Copyright 2026 The DLRover Authors. All rights reserved.
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

from kubernetes import client, config, watch
import time
from typing import Callable
from dlrover.brain.python.common.log import default_logger as logger


class ConfigMapWatcher:
    def __init__(
        self, namespace, name, on_update_callback: Callable[[dict], None]
    ):
        self.name = name
        self.namespace = namespace
        self.on_update_callback = on_update_callback

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()  # Fallback for local testing

        self.v1 = client.CoreV1Api()

    def watch(self):
        resource_version = None

        while True:
            try:
                w = watch.Watch()

                stream_kwargs = {
                    "namespace": self.namespace,
                    "field_selector": f"metadata.name={self.name}",
                    "timeout_seconds": 0,  # 0 means "listen as long as possible"
                }

                if resource_version:
                    stream_kwargs["resource_version"] = resource_version

                for event in w.stream(
                    self.v1.list_namespaced_config_map, **stream_kwargs
                ):
                    obj = event["object"]
                    event_type = event["type"]

                    resource_version = obj.metadata.resource_version

                    if event_type == "MODIFIED" or event_type == "ADDED":
                        if obj.data:
                            print(f"data: {obj.data}")
                            self.on_update_callback(obj.data)

            except Exception as e:
                logger.warning(
                    f"Watch connection broken ({e}). Retrying in 5 seconds..."
                )
                time.sleep(5)
