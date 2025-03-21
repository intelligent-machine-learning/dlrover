# Copyright 2024 The DLRover Authors. All rights reserved.
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

import copy
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from datetime import datetime
from unittest import mock
from unittest.mock import MagicMock, patch

import psutil
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent,
)
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.launcher.api import LaunchConfig

from dlrover.python.common import env_utils
from dlrover.python.common.constants import (
    Accelerators,
    AscendConstants,
    ConfigPath,
    GpuMetricEnum,
    JobConstant,
    NodeEnv,
    NpuMetricEnum,
    RendezvousName,
)
from dlrover.python.common.global_context import Context
from dlrover.python.common.log import default_logger as logger
from dlrover.python.common.metric.context import JobMetricContext
from dlrover.python.common.metric.metric import (
    GpuMetric,
    GpuNodeMetric,
    NpuMetric,
    NpuNodeMetric,
)
from dlrover.python.common.storage import PosixDiskStorage
from dlrover.python.diagnosis.common.constants import DiagnosisConstant
from dlrover.python.diagnosis.common.diagnosis_action import EventAction
from dlrover.python.elastic_agent.context import get_agent_context
from dlrover.python.elastic_agent.master_client import (
    MasterClient,
    build_master_client,
)
from dlrover.python.elastic_agent.monitor.training import TorchTrainingMonitor
from dlrover.python.elastic_agent.torch.ckpt_saver import (
    AsyncCheckpointSaver,
    DdpCheckpointSaver,
)
from dlrover.python.elastic_agent.torch.training import (
    ElasticLaunchConfig,
    ElasticTrainingAgent,
    JobStoppingError,
    MasterRendezvousHandler,
    NodeCheckElasticAgent,
    NodeCheckFailedError,
    RendezvousOutSyncError,
    RendezvousTimeoutError,
    StopWorkerTimeoutError,
    _create_check_agent,
    _create_worker_spec,
    _get_local_ip,
    _set_paral_config,
    comm_perf_check,
    launch_agent,
    node_health_check,
)
from dlrover.python.tests.test_utils import start_local_master

_metric_context = JobMetricContext.singleton_instance()
_dlrover_context = Context.singleton_instance()


class ElasticTrainingAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        _set_paral_config()
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 0.5)
        launch_config = LaunchConfig(
            min_nodes=2,
            max_nodes=2,
            nproc_per_node=8,
            run_id="test",
        )
        self.config = ElasticLaunchConfig(**launch_config.__dict__)
        self.config.set_node_unit(2)
        rdzv_parameters = RendezvousParameters(
            backend=self.config.rdzv_backend,
            endpoint=self.config.rdzv_endpoint,
            run_id=self.config.run_id,
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            local_addr=self.config.local_addr,
            **self.config.rdzv_configs,
        )

        master_addr = "127.0.0.1"

        self.rdzv_handler = MasterRendezvousHandler(
            RendezvousName.ELASTIC_TRAINING,
            0,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        self.rdzv_handler.join_timeout = 5

        self.spec = WorkerSpec(
            role=self.config.role,
            local_world_size=self.config.nproc_per_node,
            entrypoint="echo",
            args=tuple([]),
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            redirects=self.config.redirects,
            tee=self.config.tee,
            master_addr=master_addr,
            local_addr=self.config.local_addr,
        )
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 1

    def tearDown(self):
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 15
        self._master.stop()
        os.environ.clear()

    def test_node_unit(self):
        node_unit = int(self.rdzv_handler._rdzv_params.get("node_unit", "1"))
        self.assertEqual(node_unit, 2)

    def test_config_to_json(self):
        config = ElasticLaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=8,
            run_id="test",
            auto_config=True,
        )
        json_config = config.to_json()
        self.assertIsNotNone(json_config)
        self.assertEqual(json_config["min_nodes"], 1)
        self.assertEqual(json_config["rdzv_backend"], "etcd")
        self.assertTrue(len(json.loads(json_config["rdzv_configs"])) > 0)

    def test_auto_configure(self):
        config = ElasticLaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=8,
            run_id="test",
            auto_config=True,
        )
        os.environ["NODE_NUM"] = "4"
        os.environ["TRAINING_LOG_FILE"] = "training_log"
        os.environ["FAILURE_NODE_ERRORS"] = "#errors#"
        config.auto_configure_params()
        self.assertEqual(config.max_nodes, 4)
        self.assertEqual(config.min_nodes, 4)
        self.assertTrue(config.network_check)
        self.assertEqual(config.training_log_file, "training_log")
        self.assertEqual(config.failure_node_errors, "#errors#")

        os.environ["FAILURE_NODE_ERRORS"] = " #errors"
        config.auto_configure_params()
        self.assertEqual(config.failure_node_errors, "")

    def test_rank0_rendezvous(self):
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="python",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )

        # Mock node rank 1 joins the rendezvous.
        self.rdzv_handler._client._node_id = 1
        self.rdzv_handler._client.join_rendezvous(
            1, 8, self.rdzv_handler._name
        )
        agent._client._node_id = 0
        agent._rendezvous(agent._worker_group)
        worker_group = agent._worker_group
        self.assertEqual(len(worker_group.workers), 8)
        self.assertEqual(worker_group.group_rank, 0)
        self.assertEqual(worker_group.group_world_size, 2)
        worker = worker_group.workers[1]
        self.assertEqual(worker.local_rank, 1)
        self.assertEqual(worker.global_rank, 1)
        self.assertEqual(worker.world_size, 16)
        self.assertFalse(
            agent._membership_changed("default", self.rdzv_handler)
        )

    def test_rank1_rendezvous(self):
        agent = ElasticTrainingAgent(
            node_rank=1,
            config=self.config,
            entrypoint="python",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        # Mock node rank 0 joins the rendezvous.
        self.rdzv_handler._client._node_id = 0
        self.rdzv_handler._client.join_rendezvous(
            0, 8, self.rdzv_handler._name
        )

        store = self.rdzv_handler._get_store(round=1, group=0)

        def _set_store(store):
            time.sleep(1)
            store.set("MASTER_ADDR", "127.0.0.1".encode())
            store.set("MASTER_PORT", "12345".encode())

        _task = threading.Thread(target=_set_store, args=(store,))
        _task.start()

        addr, port = agent._safe_get_master_addr_port(store)
        self.assertEqual(addr, "127.0.0.1")
        self.assertEqual(port, 12345)

        # Set the node id and rank as 1.
        agent._client._node_id = 1
        self.spec.rdzv_handler._node_rank = 1
        agent._rendezvous(agent._worker_group)
        worker_group = agent._worker_group
        self.assertEqual(len(worker_group.workers), 8)
        self.assertEqual(worker_group.group_rank, 1)
        self.assertEqual(worker_group.group_world_size, 2)
        worker = worker_group.workers[1]
        self.assertEqual(worker.local_rank, 1)
        self.assertEqual(worker.global_rank, 9)
        self.assertEqual(worker.world_size, 16)
        self.assertEqual(store.get("MASTER_ADDR").decode(), "127.0.0.1")
        self.assertEqual(store.get("MASTER_PORT").decode(), "12345")

    def test_exit_barrier(self):
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="python",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
            exit_barrier_timeout=1,
        )
        self.rdzv_handler._client._node_id = 1
        self.rdzv_handler._client.join_rendezvous(
            1, 8, self.rdzv_handler._name
        )
        agent._client._node_id = 0
        agent._rendezvous(agent._worker_group)
        agent._exit_barrier()

    def test_get_local_ip(self):
        local_ip = _get_local_ip()
        self.assertNotEqual(local_ip, "")
        os.environ["POD_IP"] = "127.0.0.1"
        local_ip = _get_local_ip()
        self.assertEqual(local_ip, "127.0.0.1")

    def test_initialize_worker(self):
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 1
        node_id = 1
        agent = ElasticTrainingAgent(
            node_rank=node_id,
            config=self.config,
            entrypoint="python",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        agent._config.network_check = False
        agent._config.rdzv_configs = {"pend_timeout": 0}

        def _mock_rendezvous(self, *args):
            raise RendezvousOutSyncError("test")

        agent._rendezvous = _mock_rendezvous
        with self.assertRaises(RendezvousTimeoutError):
            agent._initialize_workers(agent._worker_group)
            agent._save_ckpt_future

    def test_initialize_workers_exception(self):
        node_id = 2
        agent = ElasticTrainingAgent(
            node_rank=node_id,
            config=self.config,
            entrypoint="python",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        agent._config.network_check = False

        agent._rendezvous = mock.MagicMock(
            side_effect=[NodeCheckFailedError("test")],
        )
        with self.assertRaises(NodeCheckFailedError):
            agent._initialize_workers(agent._worker_group)

        agent._rendezvous = mock.MagicMock(
            side_effect=[RendezvousTimeoutError("test")],
        )
        with self.assertRaises(RendezvousTimeoutError):
            agent._initialize_workers(agent._worker_group)

        agent._rendezvous = mock.MagicMock(
            side_effect=[ValueError("test")],
        )
        with self.assertRaises(ValueError):
            agent._initialize_workers(agent._worker_group, max_errors=1)

        agent._rendezvous = mock.MagicMock(
            side_effect=[ValueError("test"), TypeError("test")],
        )
        with self.assertRaises(TypeError):
            agent._initialize_workers(agent._worker_group, max_errors=2)

        agent._rendezvous = mock.MagicMock(
            side_effect=[
                ValueError("test"),
                TypeError("test"),
                RuntimeError("test"),
            ],
        )
        with self.assertRaises(RuntimeError):
            agent._initialize_workers(agent._worker_group, max_errors=3)


def mock_gpu_metric_collect(*args, **kwargs):
    logger.info("mock gpu metric collector is running...")
    job_metrics = {}
    metric = GpuNodeMetric()
    for i in range(8):
        metric.node_metrics[i] = GpuMetric()
        metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_FREE_MEM, 0)
        metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_USED_MEM, 80)
        metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_UTIL, 99.5)
        metric.node_metrics[i].set_metric(GpuMetricEnum.GPU_TENSOR_UTIL, 30.5)
    metric.update_avg_metrics()
    job_metrics["worker-1"] = copy.deepcopy(metric)
    job_metrics["worker-2"] = copy.deepcopy(metric)
    job_metrics["worker-3"] = copy.deepcopy(metric)
    job_metrics["worker-4"] = copy.deepcopy(metric)
    _metric_context.add_node_metrics(
        int(datetime.now().timestamp()), job_metrics
    )


def mock_npu_metric_collect(*args, **kwargs):
    logger.info("mock npu metric collector is running...")
    job_metrics = {}
    metric = NpuNodeMetric()
    for i in range(16):
        metric.node_metrics[i] = NpuMetric()
        metric.node_metrics[i].set_metric(NpuMetricEnum.NPU_USED_MEM, 78)
        metric.node_metrics[i].set_metric(NpuMetricEnum.NPU_TOTAL_MEM, 80)
        metric.node_metrics[i].set_metric(NpuMetricEnum.NPU_UTIL, 99.5)
    metric.update_avg_metrics()
    job_metrics["worker-1"] = copy.deepcopy(metric)
    job_metrics["worker-2"] = copy.deepcopy(metric)
    job_metrics["worker-3"] = copy.deepcopy(metric)
    job_metrics["worker-4"] = copy.deepcopy(metric)
    _metric_context.add_node_metrics(
        int(datetime.now().timestamp()), job_metrics
    )


class ElasticTrainingAgentRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 1)
        self._client = MasterClient.singleton_instance()
        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            run_id="test",
            monitor_interval=0.1,
        )
        self.config = ElasticLaunchConfig(**launch_config.__dict__)
        rdzv_parameters = RendezvousParameters(
            backend=self.config.rdzv_backend,
            endpoint=self.config.rdzv_endpoint,
            run_id=self.config.run_id,
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            local_addr=self.config.local_addr,
            **self.config.rdzv_configs,
        )

        master_addr = "127.0.0.1"
        node_id = 0

        self.rdzv_handler = MasterRendezvousHandler(
            RendezvousName.ELASTIC_TRAINING,
            node_id,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        self.rdzv_handler.join_timeout = 5

        self.spec = WorkerSpec(
            role=self.config.role,
            local_world_size=self.config.nproc_per_node,
            entrypoint="echo",
            args=tuple([]),
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            redirects=self.config.redirects,
            tee=self.config.tee,
            master_addr=master_addr,
            local_addr=self.config.local_addr,
        )
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 1

    def tearDown(self):
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 15
        self._master.stop()

    def test_monitor_workers(self):
        self.config.network_check = False
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        agent._report_failure_to_master({})
        run_result = agent._invoke_run()
        self.assertDictEqual(run_result.failures, {})
        self.assertEqual(run_result.state, WorkerState.SUCCEEDED)

    @mock.patch.object(ElasticTrainingAgent, "_restart_workers")
    def test_invoke_run(self, mock_restart_workers):
        self.config.network_check = False
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        agent._monitor_workers = MagicMock(
            return_value=RunResult(
                state=WorkerState.HEALTHY,
                return_values={0: 1, 1: 0},
                failures={},
            )
        )
        agent._membership_changed = MagicMock(return_value=True)
        mock_restart_workers.side_effect = RuntimeError("test")
        try:
            agent._invoke_run()
        except RuntimeError:
            mock_restart_workers.assert_called()

    def test_metric_collect(self):
        with patch(
            "dlrover.python.common.metric.monitor.SimpleMetricMonitor._collector",  # noqa
            side_effect=mock_gpu_metric_collect(),
        ):
            os.environ[
                "DLROVER_METRIC_URL"
            ] = "https://metric.mock.dlrover.org"
            os.environ["DLROVER_METRIC_TOKEN"] = "0123456789"
            self.assertIsNot(os.getenv("DLROVER_METRIC_URL", ""), "")
            self.assertIsNot(os.getenv("DLROVER_METRIC_TOKEN", ""), "")

            _metric_context.clear_node_metrics()

            self._master.diagnosis_manager.stop_metric_collect()

        with patch(
            "dlrover.python.common.metric.monitor.SimpleMetricMonitor._collector",  # noqa
            side_effect=mock_npu_metric_collect(),
        ):
            os.environ[
                "DLROVER_METRIC_URL"
            ] = "https://metric.mock.dlrover.org"
            os.environ["DLROVER_METRIC_TOKEN"] = "0123456789"
            self.assertIsNot(os.getenv("DLROVER_METRIC_URL", ""), "")
            self.assertIsNot(os.getenv("DLROVER_METRIC_TOKEN", ""), "")

            _metric_context.clear_node_metrics()

            self._master.diagnosis_manager.stop_metric_collect()

    def test_failure_ending_after_training(self):
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        agent._wait_async_saver = mock.MagicMock(side_effect=[Exception])
        run_result = agent._invoke_run()
        self.assertDictEqual(run_result.failures, {})
        self.assertEqual(run_result.state, WorkerState.SUCCEEDED)

    def test_report_step(self):
        os.environ[NodeEnv.MONITOR_ENABLED] = "true"
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_file = os.path.join(tmpdirname, "runtime_metrics.json")
            monitor = TorchTrainingMonitor(config_file)
            monitor.start()
            monitor.report_step()
            self.assertEqual(self._master.perf_monitor._global_step, 0)
            record = {"step": 100, "timestamp": time.time()}
            with open(config_file, "w") as f:
                f.write(json.dumps(record))

            monitor.report_step()
            self.assertEqual(self._master.perf_monitor._global_step, 100)

    def test_check_network_rdzv(self):
        self._master.rdzv_managers[
            RendezvousName.NETWORK_CHECK
        ].join_rendezvous(0, 0, 8)
        with self.assertRaises(RendezvousOutSyncError):
            self.rdzv_handler._check_network_rdzv()

    def test_get_free_port(self):
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )

        os.environ["HOST_PORTS"] = "10000,10002,10003"
        port = agent._get_free_port()
        self.assertTrue(port in [10000, 10002, 10003])

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 10000))
        os.environ["HOST_PORTS"] = "10000"
        port = agent._get_free_port()
        s.close()
        self.assertTrue(port != 10000)

        os.environ["HOST_PORTS"] = ""
        port = agent._get_free_port()
        self.assertTrue(port > 20000)

    def test_restart_training(self):
        self.config.restart = True
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        storage = PosixDiskStorage()
        saver = DdpCheckpointSaver("/tmp/test", storage.get_class_meta())
        AsyncCheckpointSaver._saver_instance = saver
        agent._save_ckpt_to_storage()
        agent._stop_workers_to_restart()
        agent._wait_async_saver()
        agent.stop_executor()

    def test_create_worker_spec(self):
        spec = _create_worker_spec(
            node_rank=0,
            rdzv_name=RendezvousName.ELASTIC_TRAINING,
            config=self.config,
            entrypoint="echo",
            args=[],
        )
        self.assertEqual(spec.max_restarts, 3)
        self.assertEqual(spec.local_world_size, 2)

    def test_numa_affinity(self):
        with patch(
            "dlrover.python.util.numa_util.get_npu_affinity",
            return_value={0, 1},
        ):
            self.config.numa_affinity = True
            self.config.training_port = 0
            self.config.accelerator = Accelerators.ASCEND_NPU
            self.spec.entrypoint = "sleep"
            self.spec.args = tuple(["3"])
            agent = ElasticTrainingAgent(
                node_rank=0,
                config=self.config,
                entrypoint="sleep",
                spec=self.spec,
                start_method=self.config.start_method,
                log_dir=self.config.log_dir,
            )
            self.assertEqual(agent._rank_cpu_affinity[0], None)
            self.assertEqual(agent._rank_cpu_affinity[1], None)
            agent._rank_cpu_affinity[0] = {0, 1}
            agent._rank_cpu_affinity[1] = {2, 3}
            run_result = agent._invoke_run()
            self.assertDictEqual(run_result.failures, {})
            self.assertEqual(run_result.state, WorkerState.SUCCEEDED)

        with patch(
            "dlrover.python.util.numa_util.get_gpu_affinity",
            return_value={0, 1},
        ):
            self.config.numa_affinity = True
            self.config.accelerator = Accelerators.NVIDIA_GPU
            self.spec.entrypoint = "sleep"
            self.spec.args = tuple(["3"])
            agent = ElasticTrainingAgent(
                node_rank=0,
                config=self.config,
                entrypoint="sleep",
                spec=self.spec,
                start_method=self.config.start_method,
                log_dir=self.config.log_dir,
            )
            self.assertEqual(agent._rank_cpu_affinity[0], None)
            self.assertEqual(agent._rank_cpu_affinity[1], None)
            agent._rank_cpu_affinity[0] = {0, 1}
            agent._rank_cpu_affinity[1] = {2, 3}
            run_result = agent._invoke_run()
            self.assertDictEqual(run_result.failures, {})
            self.assertEqual(run_result.state, WorkerState.SUCCEEDED)

    def test_sync_node_port(self):
        self.config.accelerator = Accelerators.ASCEND_NPU
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        agent.sync_training_ports(1)
        self.assertEqual(
            os.environ[AscendConstants.HCCL_PORT_START],
            str(AscendConstants.HCCL_PORT_START_DEFAULT),
        )

    def test_sync_node_port_with_env(self):
        os.environ[AscendConstants.HCCL_PORT_START] = "65000"
        self.config.accelerator = Accelerators.ASCEND_NPU
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )

        agent.sync_training_ports(1)
        self.assertEqual(
            os.environ[AscendConstants.HCCL_PORT_START],
            str(65000),
        )

    def test_stop_workers_ascend(self, cmdline=None):
        # test Ascend NPU
        config = self.config
        spec = self.spec

        self.config.accelerator = Accelerators.ASCEND_NPU
        self.spec.max_restarts = 0
        if cmdline is None:
            self.spec.entrypoint = "sleep"
            self.spec.args = tuple(["180"])
        else:
            self.spec.entrypoint = cmdline[0]
            self.spec.args = tuple(cmdline[1:])

        self.config.network_check = False
        self.config.training_port = 0
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint=self.spec.entrypoint,
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )

        def stop_task(agent):
            time.sleep(1)
            agent._stop_workers_ascend(None)

        stop_task = threading.Thread(target=stop_task, args=(agent,))
        stop_task.start()

        run_result = agent._invoke_run()
        self.assertEqual(run_result.state, WorkerState.FAILED)

        stop_task.join()

        self.spec = spec
        self.config = config

    def test_no_orphan_workers(self):
        orphan_killed = True
        orphan_pid = -1
        subprocess.run(
            ["/usr/local/bin/python", "dlrover/python/tests/orphan_process.py"]
        )
        env_utils.print_process_list()
        for p in psutil.process_iter():
            try:
                self.assertIsNotNone(env_utils.get_proc_env(p.pid))
                self.assertFalse(env_utils.is_worker_process(p.pid))
            except Exception:
                pass
        self.assertIsNone(env_utils.get_proc_env(999999))

        self.test_stop_workers_ascend()

        for p in psutil.process_iter():
            try:
                name = " ".join(p.cmdline())
                if "orphan_process.py" in name:
                    orphan_killed = False
                    orphan_pid = p.pid
                    break
            except Exception:
                pass

        self.assertFalse(orphan_killed)
        os.kill(orphan_pid, signal.SIGTERM)

    def test_orphan_workers(self):
        orphan_killed = True
        subprocess.run(
            [
                "/usr/local/bin/python",
                "dlrover/python/tests/orphan_process.py",
                "torch",
            ]
        )
        env_utils.print_process_list()
        for p in psutil.process_iter():
            try:
                self.assertIsNotNone(env_utils.get_proc_env(p.pid))
                name = " ".join(p.cmdline())
                if "orphan_process.py" in name:
                    self.assertTrue(env_utils.is_worker_process(p.pid))
                else:
                    self.assertFalse(env_utils.is_worker_process(p.pid))
            except Exception:
                pass
        self.assertIsNone(env_utils.get_proc_env(999999))

        self.test_stop_workers_ascend()

        for p in psutil.process_iter():
            try:
                name = " ".join(p.cmdline())
                if "orphan_process.py" in name:
                    orphan_killed = False
                    break
            except Exception:
                pass

        self.assertTrue(orphan_killed)

    def test_stop_workers(self):
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )

        # without timeout
        agent._stop_workers(None, is_restart=False, timeout=3)

        def sleep_10_seconds(*args, **kwargs):
            time.sleep(10)

        # with timeout
        with patch.object(
            LocalElasticAgent, "_stop_workers", side_effect=sleep_10_seconds
        ):
            agent = ElasticTrainingAgent(
                node_rank=0,
                config=self.config,
                entrypoint="echo",
                spec=self.spec,
                start_method=self.config.start_method,
                log_dir=self.config.log_dir,
            )
            try:
                agent._stop_workers(None, is_restart=False, timeout=3)
                self.fail()
            except StopWorkerTimeoutError:
                self.assertTrue(True)

    def test_diagnosis(self):
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )

        context = get_agent_context()
        action = EventAction(
            event_action="action",
            expired_time_period=600,
        )
        context.enqueue_diagnosis_action(action)

        time.sleep(3)
        agent._check_and_process_diagnosis_action()
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.MASTER_INSTANCE
                ]
            ),
            1,
        )

    @patch(
        "dlrover.python.elastic_agent.master_client"
        ".MasterClient.report_failed_exited"
    )
    @patch(
        "dlrover.python.elastic_agent.torch.training"
        ".ElasticTrainingAgent.run"
    )
    def test_node_status_report(self, mock_run, mock_report_failed_exited):
        config = ElasticLaunchConfig(1, 1, 1)
        entrypoint = "python"

        mock_run.side_effect = RuntimeError("test")
        mock_report_failed_exited.return_value = True
        try:
            launch_agent(config, entrypoint, [])
            self.fail()
        except RuntimeError:
            self.assertTrue(True)
            mock_run.assert_called_once()
            mock_report_failed_exited.assert_called_once()

        mock_run.side_effect = NodeCheckFailedError("test")
        try:
            launch_agent(config, entrypoint, [])
            self.fail()
        except NodeCheckFailedError:
            self.assertTrue(True)
            self.assertEqual(mock_run.call_count, 2)
            mock_report_failed_exited.assert_called_once()

    @patch(
        "dlrover.python.elastic_agent.torch.training"
        ".ElasticTrainingAgent.run"
    )
    def test_launch_agent(self, mock_run):
        config = ElasticLaunchConfig(1, 1, 1)
        entrypoint = "python"
        mock_run.return_value = None
        try:
            launch_agent(config, entrypoint, [])
        except Exception:
            pass

        mock_run.return_value = RunResult(
            state=WorkerState.FAILED,
            return_values={0: 1, 1: 0},
            failures={},
        )
        try:
            launch_agent(config, entrypoint, [])
        except Exception:
            pass


class NodeCheckElasticAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 0.5)
        launch_config = LaunchConfig(
            min_nodes=2,
            max_nodes=2,
            nproc_per_node=8,
            run_id="test",
        )
        self.config = ElasticLaunchConfig(**launch_config.__dict__)
        rdzv_parameters = RendezvousParameters(
            backend=self.config.rdzv_backend,
            endpoint=self.config.rdzv_endpoint,
            run_id=self.config.run_id,
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            local_addr=self.config.local_addr,
            **self.config.rdzv_configs,
        )

        master_addr = "127.0.0.1"
        node_id = 0

        self.rdzv_handler = MasterRendezvousHandler(
            RendezvousName.ELASTIC_TRAINING,
            node_id,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        self.rdzv_handler.join_timeout = 5

        self.spec = WorkerSpec(
            role=self.config.role,
            local_world_size=self.config.nproc_per_node,
            entrypoint="echo",
            args=tuple([]),
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            redirects=self.config.redirects,
            tee=self.config.tee,
            master_addr=master_addr,
            local_addr=self.config.local_addr,
        )
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 1

    def tearDown(self):
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 15
        self._master.stop()

    def test_get_network_check_time(self):
        node_id = 0
        agent = NodeCheckElasticAgent(
            node_rank=node_id,
            config=self.config,
            entrypoint="python",
            spec=self.spec,
            start_method=self.config.start_method,
            log_dir=self.config.log_dir,
        )
        root = ConfigPath.NETWORK_CHECK_DATA_DIR
        if os.path.exists(root):
            shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)
        for i in range(8):
            data = {"rank": i, "time": 100 + i}
            path = os.path.join(root, f"{i}.json")
            with open(path, "w") as f:
                f.write(json.dumps(data))
        finished = agent._check_finished(root)
        self.assertTrue(finished)
        t = agent._get_node_check_time(root)
        self.assertEqual(t, 107)
        if os.path.exists(root):
            shutil.rmtree(root)

    def test_create_check_agent(self):
        config = ElasticLaunchConfig(4, 4, 8)
        agent = _create_check_agent(
            config=config,
            entrypoint="python",
            args=[],
            rdzv_name="elastic-training",
            check_round=2,
        )
        self.assertEqual(agent._check_round, 2)

    def test_run_agent(self):
        config = ElasticLaunchConfig(4, 4, 8)
        agent = _create_check_agent(
            config=config,
            entrypoint="python",
            args=[],
            rdzv_name="elastic-training",
            check_round=2,
        )

        # with no fault and no stragglers
        agent._client.check_fault_node = mock.MagicMock(return_value=([], ""))
        agent._client.check_straggler = mock.MagicMock(return_value=([], ""))
        agent._run_node_check = mock.MagicMock(return_value=(True, 100))
        agent._stop_workers = mock.MagicMock(return_value=True)
        agent._client.report_network_check = mock.MagicMock(return_value=True)
        self.assertTrue(agent.run())

        # with fault and no stragglers
        agent._client.check_fault_node = mock.MagicMock(return_value=([0], ""))
        agent._client.check_straggler = mock.MagicMock(return_value=([], ""))
        try:
            agent.run()
            self.fail()
        except RuntimeError:
            pass

        # with no fault and stragglers
        agent._client.check_fault_node = mock.MagicMock(return_value=([], ""))
        agent._client.check_straggler = mock.MagicMock(return_value=([0], ""))
        self.assertTrue(agent.run())

        # with fault and stragglers
        agent._client.check_fault_node = mock.MagicMock(return_value=([1], ""))
        agent._client.check_straggler = mock.MagicMock(return_value=([0], ""))
        try:
            agent.run()
            self.fail()
        except RuntimeError:
            pass

        # with _run_node_check return false
        agent._client.check_fault_node = mock.MagicMock(return_value=([], ""))
        agent._client.check_straggler = mock.MagicMock(return_value=([], ""))
        agent._run_node_check = mock.MagicMock(return_value=(False, 100))
        self.assertFalse(agent.run())

    @mock.patch.object(NodeCheckElasticAgent, "run")
    def test_node_health_check(self, mock_run):
        config = ElasticLaunchConfig(1, 1, 1)
        entrypoint = "python"
        args = "--version"
        node_health_check(config, entrypoint, args)
        mock_run.assert_called()

    @mock.patch.object(NodeCheckElasticAgent, "run")
    def test_comm_perf_test(self, mock_run):
        config = ElasticLaunchConfig(1, 1, 1)
        entrypoint = "python"
        args = "--version"
        comm_perf_check(config, entrypoint, args)
        mock_run.assert_called()

    def test_get_check_node_timeout(self):
        config = ElasticLaunchConfig(4, 4, 8)

        agent = _create_check_agent(
            config=config,
            entrypoint="python",
            args=[],
            rdzv_name="elastic-training",
            check_round=2,
        )
        self.assertEqual(
            agent._get_check_node_timeout(),
            JobConstant.MASTER_CLIENT_CHECK_NODE_TIMEOUT,
        )


class MasterRendezvousHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 0.5)
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 1

    def tearDown(self):
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 15
        self._master.stop()

    def test_join_rendezvous(self):
        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            run_id="test",
            monitor_interval=0.1,
        )
        self.config = ElasticLaunchConfig(**launch_config.__dict__)
        rdzv_parameters = RendezvousParameters(
            backend=self.config.rdzv_backend,
            endpoint=self.config.rdzv_endpoint,
            run_id=self.config.run_id,
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            local_addr=self.config.local_addr,
            **self.config.rdzv_configs,
        )
        rdzv_handler = MasterRendezvousHandler(
            RendezvousName.ELASTIC_TRAINING,
            0,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        rdzv_handler._client.join_rendezvous = mock.MagicMock(return_value=0)
        rdzv_handler._client.num_nodes_waiting = mock.MagicMock(
            return_value=-1
        )
        with self.assertRaises(JobStoppingError):
            rdzv_handler.next_rendezvous()

        rdzv_handler._client.join_rendezvous = mock.MagicMock(return_value=0)
        rdzv_handler._client.num_nodes_waiting = mock.MagicMock(return_value=1)
        with self.assertRaises(RendezvousOutSyncError):
            rdzv_handler.next_rendezvous()

    def test_pend_timeout(self):
        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            run_id="test",
            monitor_interval=0.1,
        )
        self.config = ElasticLaunchConfig(**launch_config.__dict__)
        rdzv_parameters = RendezvousParameters(
            backend=self.config.rdzv_backend,
            endpoint=self.config.rdzv_endpoint,
            run_id=self.config.run_id,
            min_nodes=self.config.min_nodes,
            max_nodes=self.config.max_nodes,
            local_addr=self.config.local_addr,
            **self.config.rdzv_configs,
        )
        rdzv_parameters.config["pend_timeout"] = 1
        rdzv_handler = MasterRendezvousHandler(
            RendezvousName.ELASTIC_TRAINING,
            0,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        rdzv_handler._join_rendezvous = mock.MagicMock(return_value=0)
        rdzv_handler._client.get_comm_world = mock.MagicMock(
            return_value=(0, 0, {1: 8})
        )
        with self.assertRaises(RendezvousTimeoutError):
            rdzv_handler.next_rendezvous()


if __name__ == "__main__":
    unittest.main()
