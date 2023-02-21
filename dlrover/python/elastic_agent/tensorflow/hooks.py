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

import tensorflow as tf
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import (
    SessionRunArgs,
    SessionRunHook,
)

from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import GlobalMasterClient
from dlrover.python.elastic_agent.monitor.training import (
    TrainingProcessReporter,
    is_tf_chief,
)
from dlrover.python.elastic_agent.sharding.client import ShardingClient
from dlrover.python.elastic_agent.tensorflow.profile_extractor import (
    OperationStats,
    TensorStats,
)

training_reporter = TrainingProcessReporter()


def collect_model_stats(flops):
    tensor_stats, op_stats = generate_model_stats()
    op_stats.runtime_flops = flops
    GlobalMasterClient.MASTER_CLIENT.report_model_metric(
        tensor_stats,
        op_stats,
    )


def generate_model_stats():
    op_stats = OperationStats()
    tensor_stats = TensorStats()
    all_ops = tf.get_default_graph().get_operations()
    op_stats.op_count = len(all_ops)
    for op in all_ops:
        if "update_" in op.name:  # Ops with update_ executed on PS
            op_stats.update_op_count += 1
        if op.name.endswith("/read") or op.name.endswith(
            "/Read/ReadVariableOp"
        ):
            op_stats.read_op_count += 1
    tensor_stats.update_varible_stats(tf.global_variables())
    return tensor_stats, op_stats


class ReportModelMetricHook(SessionRunHook):
    def __init__(self):
        """Report variables and operators in a model to
        the DLRover master.
        """
        self._is_chief = False
        training_reporter.called_in_tf_hook = True
        self._global_step = 0
        self._op_stats = None
        self._tensor_stats = None
        super(ReportModelMetricHook, self).__init__()

    def begin(self):
        self._is_chief = is_tf_chief()
        if not self._is_chief:
            return
        _create_fn = training_util._get_or_create_global_step_read
        self._global_step_tensor = _create_fn()

    def after_create_session(self, session, coord):
        if not self._is_chief:
            return
        try:
            tensor_stats, op_stats = generate_model_stats()
            GlobalMasterClient.MASTER_CLIENT.report_model_metric(
                tensor_stats,
                op_stats,
            )
            training_reporter.set_start_time()
        except Exception as e:
            logger.warning(e)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        if not self._is_chief:
            return
        requests = {"global_step": self._global_step_tensor}
        return SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        if not self._is_chief:
            return
        self._global_step = run_values.results["global_step"]
        training_reporter.report_resource_with_step(self._global_step)


class ElasticDataShardReportHook(SessionRunHook):
    def __init__(self, sharding_client: ShardingClient) -> None:
        self._sharding_client = sharding_client

    def after_run(self, run_context, run_values):
        try:
            self._sharding_client.report_batch_done()
        except Exception as ex:
            logger.error("DLrover agent: report batch done failed: %s", ex)
