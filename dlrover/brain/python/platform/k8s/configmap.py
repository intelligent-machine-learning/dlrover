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
import time
import json
from kubernetes import client, config, watch
from typing import Callable
from dlrover.brain.python.common.log import default_logger as logger

JSON_KEY = "config.json"

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

class ConfigMapReader:
    def __init__(self, namespace, name):
        self.name = name
        self.namespace = namespace
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        self.v1 = client.CoreV1Api()

    def read(self):

        try:
            configmap = self.v1.read_namespaced_config_map(self.name, self.namespace)

            data = configmap.data
            if not data:
                logger.warning(f"ConfigMap {self.name} has no data")
                return None
            return data
        except Exception as e:
            logger.error(f"Failed to read ConfigMap {self.name}: {e}")
            return None

    def read_json_Data(self):

        try:
            configmap = self.v1.read_namespaced_config_map(self.name, self.namespace)

            data = configmap.data
            if not data:
                logger.warning(f"ConfigMap {self.name} has no data")
                return None
        
            if JSON_KEY not in data:
                logger.warning(f"The key {JSON_KEY} was not found in the ConfigMap")
                return None
            json_str_content = data[JSON_KEY]
            
            opt_config = json.loads(json_str_content)
            if opt_config is None:
                logger.warning(f"opt_config is None")
                return None
            return opt_config
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed，please check the format of the ConfigMap content：{e}")
            return None
        except Exception as e:
            logger.error(f"Unknown error, failed to read ConfigMap {self.name}: {e}")
            return None

    def get_opt_config(self):
        try:
            configmap_reader = ConfigMapReader(self.namespace, "brain-config")
            data = configmap_reader.read_json_Data()
            if not isinstance(data, dict):
                return {}
            result = data.get("opt_config", {})
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.error(f"Failed to read opt_config from ConfigMap: {e}")
            return {}