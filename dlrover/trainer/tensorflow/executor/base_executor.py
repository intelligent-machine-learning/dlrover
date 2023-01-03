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

import json
import os

import tensorflow as tf
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.python.training import server_lib

from dlrover.trainer.util.log_util import default_logger as logger


class BaseExecutor:
    def __init__(self):
        self.cluster_spec = None
        self.task_id = None
        self.task_type = None
        self.address = None
        self.role = None
        self.cluster_spec = None
        self.mini_cluster_spec = None

    def get_cluster_info_by_tf_config(self):
        """
        get cluster info by TF_CONFIG
        {"cluster": {
                        "ps": ["web04-pod2.default.svc:5002"],
                        "chief": ["web04-pod1.default.svc:5000"]
                    },
         "task": {"type": "ps", "index": 0}}'
        """
        self.address = None
        TF_CONFIG = os.getenv("TF_CONFIG")
        if not isinstance(TF_CONFIG, str):
            logger.error("fail to parse tf config")
        TF_CONFIG = json.loads(TF_CONFIG)
        task_type = TF_CONFIG["task"]["type"]
        task_id = TF_CONFIG["task"]["index"]
        self.task_type = task_type
        self.task_id = task_id
        self.role = task_type + ":" + str(task_id)
        self.cluster_spec = TF_CONFIG["cluster"]
        if self.task_type != "evaluator":
            self.address = TF_CONFIG["cluster"][task_type][task_id]
            # self.cluster_spec = TF_CONFIG["cluster"]

            logger.info(
                "cluster spec is {} \
                    task_type is {} \
                    task_id is {} \
                    address is {}".format(
                    self.cluster_spec,
                    self.task_type,
                    self.task_id,
                    self.address,
                )
            )

    def get_cluster_def(self, cluster_spec):
        """
        {
             "ps": ["web04-pod2.default.svc:5002"],
             "chief": ["web04-pod1.default.svc:5000"],
             "worker": ["web04-pod2.default.svc:5000"],
        }
        "保留 ps"
        "保留 本地worker"
        """
        mini_cluster_spec = {}
        ps_hosts = []
        worker_hosts = []
        cluster_def = cluster_pb2.ClusterDef()
        for job_name in cluster_spec:
            if job_name == "ps":
                job = cluster_def.job.add()
                job.name = job_name
                for task_index, address in enumerate(cluster_spec[job_name]):
                    job.tasks[task_index] = address
                    ps_hosts.append(address)
            elif job_name == self.task_type:
                job = cluster_def.job.add()
                task_id = self.task_id
                if job_name == "chief":
                    job_name = "chief"
                elif job_name == "worker":
                    task_id = self.task_id + 1
                job.name = job_name
                job.tasks[task_id] = self.address
        if self.task_type != "ps":
            worker_hosts.append(self.address)
        mini_cluster_spec["ps"] = ps_hosts
        if self.task_type == "chief":
            mini_cluster_spec["chief"] = worker_hosts
        else:
            mini_cluster_spec["worker"] = worker_hosts

        if self.task_type == "chief":
            print(1)
            # mini_cluster_spec.pop("worker")
            # mini_cluster_spec["chief"] = worker_hosts

        self.mini_cluster_spec = mini_cluster_spec
        logger.info("cluster def is:\n {}".format(cluster_def))
        return cluster_def

    def start_server(self):
        if self.task_type!="evaluator":
            logger.info("starting server")
            self.server = server_lib.Server(
                {"localhost": [self.address]}, protocol="grpc"
            )
            self.server.start()

    def get_config(self, cluster_spec):
        config = tf.estimator.RunConfig()
        if self.task_type != "evaluator":
            tf_config = os.environ["TF_CONFIG"]
            tf_config = json.loads(tf_config)
            tf_config["environment"] = "google"
            os.environ["TF_CONFIG"] = json.dumps(tf_config)
            cluster_def = self.get_cluster_def(cluster_spec)
            session_config = tf.ConfigProto(
                cluster_def=cluster_def,
                gpu_options=tf.GPUOptions(allow_growth=True),
                allow_soft_placement=True,
                log_device_placement=False,
            )
            config = tf.estimator.RunConfig()
            logger.info("Using _get_run_config : %s", str(vars(config)))
            experimental_config = session_config.experimental
            experimental_config.share_session_state_in_clusterspec_propagation = (  # noqa: E501
                True
            )
            config._session_config = session_config
            config._is_chief = self.task_type == "chief"
            logger.info(
                "mini cluster spec is {}".format(self.mini_cluster_spec)
            )
            config._cluster_spec = server_lib.ClusterSpec(
                self.mini_cluster_spec
            )
            config._task_id = self.task_id
            if self.task_type == "worker":
                config._task_id = self.task_id + 1
            config._task_type = self.task_type
            if self.task_type == "chief":
                config._task_type = "chief"
            config._num_ps_replicas = len(self.mini_cluster_spec.get("ps", {}))
            config._num_worker_replicas = 1
            config._master = "grpc://" + self.address
            config._protocol = "grpc"
            config._model_dir = "model_dir"
            config._log_step_count_steps = 1e20
            config._server_name = self.address

            logger.info("config is {}".format(config))
        return config
