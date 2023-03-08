# Copyright 2020 The DLRover Authors. All rights reserved.
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

from dlrover.proto import brain_pb2, brain_pb2_grpc
from dlrover.python.common.grpc import build_channel, grpc_server_ready
from dlrover.python.common.log import default_logger as logger

DATA_STORE = "base_datastore"
OPTIMIZE_PROCESSOR = "running_training_job_optimize_request_processor"
BASE_OPTIMIZE_PROCESSOR = "base_optimize_processor"

_ENV_BRAIN_ADDR_KEY = "DLROVER_BRAIN_SERVICE_ADDR"
_DEFAULT_BRAIN_ADDR = "dlrover-brain.dlrover.svc.cluster.local:50001"


def catch_exception(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.warning(
                "Fail to %s.%s by %s",
                self.__class__.__name__,
                func.__name__,
                e,
            )

    return wrapper


def init_job_metrics_message(job_meta):
    job_metrics = brain_pb2.JobMetrics()
    job_metrics.data_store = DATA_STORE
    job_metrics.job_meta.uuid = job_meta.uuid
    job_metrics.job_meta.name = job_meta.name
    job_metrics.job_meta.user = job_meta.user
    job_metrics.job_meta.cluster = job_meta.cluster
    job_metrics.job_meta.namespace = job_meta.namespace
    return job_metrics


class JobMeta(object):
    def __init__(self, uuid, name="", namespace="", cluster="", user=""):
        self.uuid = uuid
        self.name = name
        self.namespace = namespace
        self.cluster = cluster
        self.user = user


class BrainClient(object):
    """EasyClient provides APIs to access EasyDL service via gRPC calls.

    Usage:
        channel = elasticai_api.util.grpc_utils.build_channel(
            "localhost:50001"
        )
        easydl_client = EasydlClient((channel, job_name="test")
        # Report metrics to EasyDL server
        easydl_client.report(...)
    """

    def __init__(self, brain_channel):
        """Initialize an EasyDL client.
        Args:
            channel: grpc.Channel
            the gRPC channel object connects to master gRPC server.

            job_name: string
            the unique and ordered worker ID assigned
            by dlrover command-line.
        """
        if brain_channel:
            self._brain_stub = brain_pb2_grpc.BrainStub(brain_channel)
        else:
            logger.warning("Cannot initialize brain channel")
            self._brain_stub = None

    def available(self):
        return self._brain_stub is not None

    def report_metrics(self, job_metrics):
        """Report job metrics to administer service"""
        return self._brain_stub.persist_metrics(job_metrics)

    def get_job_metrics(self, job_uuid):
        """Get the job metrics by the job uuid.
        Examples:
            ```
            import json

            client = build_brain_client()
            metrics_res = client.get_job_metrics("xxxx")
            metrics = json.loads(metrics_res.job_metrics)
            ```
        """
        request = brain_pb2.JobMetricsRequest()
        request.job_uuid = job_uuid
        return self._brain_stub.get_job_metrics(request)

    def request_optimization(self, opt_request):
        """Get the optimization plan from the processor service"""
        logger.debug("Optimization request is %s", opt_request)
        return self._brain_stub.optimize(opt_request)

    def report_training_hyper_params(self, job_meta, hyper_params):
        job_metrics = init_job_metrics_message(job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Training_Hyper_Params
        metrics = job_metrics.training_hyper_params
        metrics.batch_size = hyper_params.batch_size
        metrics.epoch = hyper_params.epoch
        metrics.max_steps = hyper_params.max_steps
        return self.report_metrics(job_metrics)

    def report_workflow_feature(self, job_meta, workflow_feature):
        job_metrics = init_job_metrics_message(job_meta)
        job_metrics.job_meta.name = workflow_feature.job_name
        job_metrics.job_meta.user = workflow_feature.user_id
        job_metrics.metrics_type = brain_pb2.MetricsType.Workflow_Feature

        metrics = job_metrics.workflow_feature
        metrics.job_name = workflow_feature.job_name
        metrics.user_id = workflow_feature.user_id
        metrics.code_address = workflow_feature.code_address
        metrics.workflow_id = workflow_feature.workflow_id
        metrics.node_id = workflow_feature.node_id
        metrics.odps_project = workflow_feature.odps_project
        metrics.is_prod = workflow_feature.is_prod
        return self.report_metrics(job_metrics)

    def report_training_set_metric(self, job_meta, dataset_metric):
        job_metrics = init_job_metrics_message(job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Training_Set_Feature
        metrics = job_metrics.training_set_feature
        metrics.dataset_size = dataset_metric.dataset_size
        metrics.dataset_name = dataset_metric.dataset_name
        sparse_features = dataset_metric.sparse_features
        metrics.sparse_item_count = sparse_features.item_count
        metrics.sparse_features = ",".join(sparse_features.feature_names)
        metrics.sparse_feature_groups = ",".join(
            [str(i) for i in sparse_features.feature_groups]
        )
        metrics.sparse_feature_shapes = ",".join(
            [str(i) for i in sparse_features.feature_shapes]
        )
        metrics.dense_features = ",".join(
            dataset_metric.dense_features.feature_names
        )
        metrics.dense_feature_shapes = ",".join(
            [str(i) for i in dataset_metric.dense_features.feature_shapes]
        )
        metrics.storage_size = dataset_metric.storage_size
        return self.report_metrics(job_metrics)

    def report_model_feature(self, job_meta, tensor_stats, op_stats):
        job_metrics = init_job_metrics_message(job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Model_Feature
        metrics = job_metrics.model_feature
        metrics.variable_count = tensor_stats.variable_count
        metrics.total_variable_size = tensor_stats.total_variable_size
        metrics.max_variable_size = tensor_stats.max_variable_size
        metrics.kv_embedding_dims.extend(tensor_stats.kv_embedding_dims)
        metrics.tensor_alloc_bytes.update(tensor_stats.tensor_alloc_bytes)
        metrics.op_count = op_stats.op_count
        metrics.update_op_count = op_stats.update_op_count
        metrics.read_op_count = op_stats.read_op_count
        metrics.input_fetch_dur = op_stats.input_fetch_dur
        metrics.flops = op_stats.flops
        metrics.recv_op_count = op_stats.recv_op_count
        return self.report_metrics(job_metrics)

    def report_runtime_info(self, job_meta, namespace, runtime_metric):
        job_metrics = init_job_metrics_message(job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Runtime_Info
        metrics = job_metrics.runtime_info
        metrics.global_step = runtime_metric.global_step
        metrics.time_stamp = runtime_metric.timestamp
        metrics.speed = runtime_metric.speed
        for pod in runtime_metric.running_nodes:
            pod_meta = brain_pb2.PodMeta()
            pod_meta.pod_name = pod.name
            pod_meta.pod_ip = pod.pod_ip
            pod_meta.node_ip = pod.node_ip
            pod_meta.host_name = pod.host_name
            pod_meta.namespace = namespace
            pod_meta.is_mixed = pod.qos == "SigmaBestEffort"
            pod_meta.mem_usage = pod.mem_usage
            pod_meta.cpu_usage = pod.cpu_usage
            metrics.running_nodes.append(pod_meta)
        return self.report_metrics(job_metrics)

    def get_optimization_plan(self, job_uuid, stage, opt_retriever, config={}):
        request = brain_pb2.OptimizeRequest()
        request.type = stage
        request.config.optimizer_config_retriever = opt_retriever
        request.config.data_store = DATA_STORE
        request.config.brain_processor = OPTIMIZE_PROCESSOR
        for key, value in config.items():
            request.config.customized_config[key] = value
        request.jobs.add()
        request.jobs[0].uid = job_uuid
        return self.request_optimization(request)

    def get_optimizer_resource_plan(self, job_uuid, opt_retriever, config={}):
        request = brain_pb2.OptimizeRequest()
        request.config.optimizer_config_retriever = opt_retriever
        request.config.data_store = DATA_STORE
        request.config.brain_processor = BASE_OPTIMIZE_PROCESSOR
        for key, value in config.items():
            request.config.customized_config[key] = value
        request.jobs.add()
        request.jobs[0].uid = job_uuid
        return self.request_optimization(request)

    def get_oom_resource_plan(
        self, nodes, job_uuid, stage, opt_retriever, config={}
    ):
        request = brain_pb2.OptimizeRequest()
        request.type = stage
        request.config.optimizer_config_retriever = opt_retriever
        request.config.data_store = DATA_STORE
        request.config.brain_processor = OPTIMIZE_PROCESSOR
        for key, value in config.items():
            request.config.customized_config[key] = value
        request.jobs.add()
        job = request.jobs[0]
        job.uid = job_uuid
        for node in nodes:
            job.state.pods[node.name].is_oom = True
            job.state.pods[node.name].name = node.name
        return self.request_optimization(request)

    def report_job_exit_reason(self, job_meta, reason):
        job_metrics = init_job_metrics_message(job_meta)
        job_metrics.metrics_type = brain_pb2.MetricsType.Job_Exit_Reason
        job_metrics.job_exit_reason = reason
        return self.report_metrics(job_metrics)

    @catch_exception
    def get_config(self, key):
        request = brain_pb2.ConfigRequest()
        request.config_key = key
        response = self._brain_stub.get_config(request)
        if response.response.success:
            return response.config_value
        return None


def build_brain_client():
    """Build a client of the EasyDL server.

    Example:
        ```
        import os
        os.environ["EASYDL_BRAIN_SERVICE_ADDR"] = "xxx"
        client = build_brain_client()
        ```
    """
    brain_addr = os.getenv(_ENV_BRAIN_ADDR_KEY, _DEFAULT_BRAIN_ADDR)
    channel = build_channel(brain_addr)
    if channel and grpc_server_ready(channel):
        return BrainClient(channel)
    else:
        logger.warning("The GRPC service of brain is not available.")
        return BrainClient(None)


class GlobalBrainClient(object):
    BRAIN_CLIENT = build_brain_client()
