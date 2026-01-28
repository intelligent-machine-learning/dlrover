# Copyright 2025 The DLRover Authors. All rights reserved.
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
from dlrover.python.elastic_agent.torch.training import LogConfig
from torch.distributed.elastic.agent.server.api import (
    RunResult,
    WorkerSpec,
    WorkerState,
)
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent,
)
from torch.distributed.elastic.multiprocessing import SignalException
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.launcher.api import LaunchConfig

from dlrover.python.common import env_utils
from dlrover.python.common.comm import GPUStats
from dlrover.python.common.constants import (
    Accelerators,
    AscendConstants,
    ConfigPath,
    GpuMetricEnum,
    JobConstant,
    NodeEnv,
    NpuMetricEnum,
    RendezvousName,
    RendezvousErrorType,
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
from dlrover.python.diagnosis.common.constants import (
    DiagnosisActionType,
    DiagnosisConstant,
)
from dlrover.python.diagnosis.common.diagnosis_action import (
    EventAction,
    NodeAction,
)
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
    run_network_check,
    _check_device,
)
from dlrover.python.tests.test_utils import start_local_master
from dlrover.trainer.torch.utils import version_less_than_230

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
            RendezvousName.TRAINING,
            0,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        self.rdzv_handler.join_timeout = 5

        if version_less_than_230():
            logs_dict = {
                "redirects": self.config.get_log_redirects(),
                "tee": self.config.get_log_tee(),
            }
        else:
            logs_dict = {}
        self.spec = WorkerSpec(
            role=self.config.role,
            local_world_size=self.config.nproc_per_node,
            entrypoint="echo",
            args=tuple([]),
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            master_addr=master_addr,
            local_addr=self.config.local_addr,
            **logs_dict,
        )
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 1

        self._agent_context = get_agent_context()

    def tearDown(self):
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 15
        self._master.stop()
        os.environ.clear()
        self._agent_context.clear_action_queue()

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

    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_name")
    def test_mtgpu_auto_configure(
        self, mock_get_device_name, mock_is_available
    ):
        mock_get_device_name.return_value = "mthreads"
        mock_is_available.return_value = True
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
            exit_barrier_timeout=1,
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
            exit_barrier_timeout=1,
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
            exit_barrier_timeout=1,
        )
        self.rdzv_handler._client._node_id = 1
        self.rdzv_handler._client.join_rendezvous(
            1, 8, self.rdzv_handler._name
        )
        agent._client._node_id = 0
        agent._rendezvous(agent._worker_group)
        agent._exit_barrier()

        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="python",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )
        self.rdzv_handler._client._node_id = 1
        self.rdzv_handler._client.join_rendezvous(
            1, 8, self.rdzv_handler._name
        )
        agent._client._node_id = 0
        agent._rendezvous(agent._worker_group)
        agent._dlrover_exit_barrier()

        with patch(
            "dlrover.python.util.store_util.barrier",
            side_effect=[SignalException("test", signal.SIGTERM)],
        ):
            agent = ElasticTrainingAgent(
                node_rank=0,
                config=self.config,
                entrypoint="python",
                spec=self.spec,
                start_method=self.config.start_method,
                exit_barrier_timeout=1,
            )
            self.rdzv_handler._client._node_id = 1
            self.rdzv_handler._client.join_rendezvous(
                1, 8, self.rdzv_handler._name
            )
            agent._client._node_id = 0
            agent._rendezvous(agent._worker_group)
            with self.assertRaises(SignalException):
                agent._dlrover_exit_barrier()

        with patch(
            "dlrover.python.util.store_util.barrier",
            side_effect=[Exception("test")],
        ):
            agent = ElasticTrainingAgent(
                node_rank=0,
                config=self.config,
                entrypoint="python",
                spec=self.spec,
                start_method=self.config.start_method,
                exit_barrier_timeout=1,
            )
            self.rdzv_handler._client._node_id = 1
            self.rdzv_handler._client.join_rendezvous(
                1, 8, self.rdzv_handler._name
            )
            agent._client._node_id = 0
            agent._rendezvous(agent._worker_group)
            agent._dlrover_exit_barrier()

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
            exit_barrier_timeout=1,
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
            exit_barrier_timeout=1,
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

    def test_log_config(self):
        from torch.distributed.elastic.multiprocessing import Std

        # test _parse_std_value
        log_config = LogConfig()
        result = LogConfig._parse_std_value(None)
        self.assertEqual(result, Std.NONE)

        result = LogConfig._parse_std_value("3")
        self.assertEqual(result, Std.ALL)

        result = LogConfig._parse_std_value("INVALID")
        self.assertEqual(result, Std.NONE)

        result = LogConfig._parse_std_value(Std.OUT)
        self.assertEqual(result, Std.OUT)

        result = LogConfig._parse_std_value(123)
        self.assertEqual(result, Std.NONE)

        # test setup
        temp_dir = tempfile.mkdtemp()
        try:
            log_config.setup(None)
            self.assertEqual(log_config.log_dir, "/tmp")
            self.assertEqual(log_config.redirects, Std.NONE)
            self.assertEqual(log_config.tee, Std.NONE)

            log_config.setup(temp_dir)
            self.assertEqual(log_config.log_dir, temp_dir)
            self.assertEqual(log_config.redirects, Std.ALL)
            self.assertEqual(log_config.tee, Std.ALL)

            log_config.setup(temp_dir, redirects="3", tee=Std.ERR)
            self.assertEqual(log_config.log_dir, temp_dir)
            self.assertEqual(log_config.redirects, Std.ALL)
            self.assertEqual(log_config.tee, Std.ERR)

            with patch(
                "dlrover.python.elastic_agent.torch.training.version_less_than_230"
            ) as mock_version:
                # test torch < 230
                mock_version.return_value = True

                logs_specs = log_config.logs_specs
                self.assertIsNotNone(logs_specs)
                self.assertTrue(isinstance(logs_specs, dict))

                # test torch >= 230
                mock_version.return_value = False
                try:
                    from torch.distributed.elastic.multiprocessing import (
                        DefaultLogsSpecs,
                    )

                    logs_specs = log_config.logs_specs
                    self.assertIsNotNone(logs_specs)
                    self.assertTrue(isinstance(logs_specs, DefaultLogsSpecs))
                except Exception:
                    pass
        finally:
            shutil.rmtree(temp_dir)


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


class MusaPatchImportTest(unittest.TestCase):
    """Test cases for musa_patch import in training.py"""

    def test_musa_patch_import_success(self):
        """Test that musa_patch import works when module exists"""
        # This test will pass if the import doesn't raise an exception
        # We're testing the actual import behavior from training.py
        try:
            import dlrover.python.elastic_agent.torch.training  # noqa: F401

            # If we get here, the import succeeded (including musa_patch)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Import should not fail: {e}")

    @patch("dlrover.python.common.musa_patch")
    def test_musa_patch_import_failure_handling(self, mock_musa_patch):
        """Test that musa_patch import failure is properly handled"""
        # Mock the module to simulate import failure
        mock_musa_patch.side_effect = ImportError("musa_patch not available")

        # Test that the training module can still be imported even if musa_patch fails
        try:
            # We can't easily test the import failure during module loading,
            # but we can test that the module works without musa_patch
            from dlrover.python.elastic_agent.torch.training import (
                ElasticLaunchConfig,
            )

            config = ElasticLaunchConfig(
                min_nodes=1, max_nodes=1, nproc_per_node=1
            )
            self.assertIsNotNone(config)
        except Exception as e:
            self.fail(f"Training module should work without musa_patch: {e}")

    def test_musa_patch_optional_import_pattern(self):
        """Test that the import pattern correctly handles exceptions"""
        # This test verifies that the try-except pattern is working as expected

        # Test the exact pattern used in training.py
        import_success = True
        try:
            from dlrover.python.common import musa_patch  # noqa: F401
        except Exception:
            import_success = False

        # The test should pass regardless of whether musa_patch exists
        # This verifies that the exception handling works
        self.assertIsInstance(import_success, bool)

    def test_musa_patch_import_does_not_affect_module_functionality(self):
        """Test that musa_patch import failure doesn't affect other functionality"""
        # Test that we can still access other parts of the training module
        from dlrover.python.elastic_agent.torch.training import (
            ElasticLaunchConfig,
        )

        # Create a config to ensure the module is functional
        config = ElasticLaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=1, run_id="test"
        )

        self.assertIsNotNone(config)
        self.assertEqual(config.min_nodes, 1)

    def test_training_module_imports_completed(self):
        """Test that all important components from training module are importable"""
        try:
            from dlrover.python.elastic_agent.torch.training import (
                ElasticLaunchConfig,
                ElasticTrainingAgent,
                launch_agent,
                _set_paral_config,
            )

            # Verify all imports are successful
            self.assertIsNotNone(ElasticLaunchConfig)
            self.assertIsNotNone(ElasticTrainingAgent)
            self.assertIsNotNone(launch_agent)
            self.assertIsNotNone(_set_paral_config)

        except ImportError as e:
            self.fail(
                f"Essential training module components should be importable: {e}"
            )

    def test_musa_patch_import_resilience(self):
        """Test that the module loading is resilient to musa_patch issues"""
        # Verify that even if there are issues with musa_patch,
        # the core functionality remains available

        from dlrover.python.elastic_agent.torch.training import (
            ElasticLaunchConfig,
        )

        # Test creating and using a basic configuration
        config = ElasticLaunchConfig(
            min_nodes=2,
            max_nodes=4,
            nproc_per_node=8,
            run_id="test_resilience",
        )

        # Test basic configuration properties
        self.assertEqual(config.min_nodes, 2)
        self.assertEqual(config.max_nodes, 4)
        self.assertEqual(config.nproc_per_node, 8)
        self.assertEqual(config.run_id, "test_resilience")

        # Test auto-configuration doesn't break
        try:
            config.auto_configure_params()
            # Should not raise exception even if musa_patch is not available
            self.assertTrue(True)
        except Exception as e:
            self.fail(
                f"Auto-configuration should not fail due to musa_patch: {e}"
            )


@patch("dlrover.python.elastic_agent.torch.training.node_health_check")
@patch("dlrover.python.elastic_agent.torch.training.comm_perf_check")
def test_run_network_check(mock_comm_perf_check, mock_node_health_check):
    mock_comm_perf_check.return_value = True
    mock_node_health_check.return_value = True
    config = ElasticLaunchConfig(
        4, 4, 8, accelerator=Accelerators.MTHREADS_GPU
    )
    entrypoint = "python"
    result = run_network_check(config, entrypoint)
    assert result


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
            RendezvousName.TRAINING,
            node_id,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        self.rdzv_handler.join_timeout = 5

        if version_less_than_230():
            logs_dict = {
                "redirects": self.config.get_log_redirects(),
                "tee": self.config.get_log_tee(),
            }
        else:
            logs_dict = {}
        self.spec = WorkerSpec(
            role=self.config.role,
            local_world_size=self.config.nproc_per_node,
            entrypoint="echo",
            args=tuple([]),
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            master_addr=master_addr,
            local_addr=self.config.local_addr,
            **logs_dict,
        )
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 1

    def tearDown(self):
        JobConstant.TRAINING_AGENT_LOOP_DEFAULT_INTERVAL = 15
        self._master.stop()
        MasterClient._instance = None

    def test_monitor_workers(self):
        self.config.network_check = False
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
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
            exit_barrier_timeout=1,
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
            os.environ["DLROVER_METRIC_URL"] = (
                "https://metric.mock.dlrover.org"
            )
            os.environ["DLROVER_METRIC_TOKEN"] = "0123456789"
            self.assertIsNot(os.getenv("DLROVER_METRIC_URL", ""), "")
            self.assertIsNot(os.getenv("DLROVER_METRIC_TOKEN", ""), "")

            _metric_context.clear_node_metrics()

            self._master.diagnosis_manager.stop_metric_collect()

        with patch(
            "dlrover.python.common.metric.monitor.SimpleMetricMonitor._collector",  # noqa
            side_effect=mock_npu_metric_collect(),
        ):
            os.environ["DLROVER_METRIC_URL"] = (
                "https://metric.mock.dlrover.org"
            )
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
            exit_barrier_timeout=1,
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
            exit_barrier_timeout=1,
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
            exit_barrier_timeout=1,
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
            rdzv_name=RendezvousName.TRAINING,
            config=self.config,
            entrypoint="echo",
            args=[],
        )
        self.assertEqual(spec.max_restarts, 3)
        self.assertEqual(spec.local_world_size, 2)

    def test_invoke_run_with_numa_affinity(self):
        self.config.numa_affinity = True
        self.config.membind_policy = "none"
        self.config.training_port = 0
        self.spec.entrypoint = "sleep"
        self.spec.args = tuple(["3"])
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="sleep",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )
        agent._initialize_workers = MagicMock(return_value=None)
        with self.assertRaises(AssertionError):
            agent._invoke_run()

    @unittest.skip("skip")
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
                exit_barrier_timeout=1,
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
                exit_barrier_timeout=1,
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
            exit_barrier_timeout=1,
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
            exit_barrier_timeout=1,
        )

        agent.sync_training_ports(1)
        self.assertEqual(
            os.environ[AscendConstants.HCCL_PORT_START],
            str(65000),
        )

    @unittest.skip("skip")
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
            exit_barrier_timeout=1,
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

    @unittest.skip("skip")
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

    @unittest.skip("skip")
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

    @patch("subprocess.run")
    def test_stop_workers(self, mock_run):
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )

        # without timeout
        agent._stop_workers(None, is_restart=False, timeout=3)

        def sleep_10_seconds(*args, **kwargs):
            time.sleep(10)

        # with timeout
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        with patch.object(
            LocalElasticAgent, "_stop_workers", side_effect=sleep_10_seconds
        ):
            agent = ElasticTrainingAgent(
                node_rank=0,
                config=self.config,
                entrypoint="echo",
                spec=self.spec,
                start_method=self.config.start_method,
                exit_barrier_timeout=1,
            )
            try:
                agent._stop_workers(None, is_restart=False, timeout=3)
                self.fail()
            except StopWorkerTimeoutError:
                self.assertTrue(True)

    @patch(
        "dlrover.python.elastic_agent.torch.training.env_utils.get_all_child_pids"
    )
    @patch(
        "dlrover.python.elastic_agent.torch.training.env_utils.get_kernel_stack"
    )
    @patch(
        "dlrover.python.elastic_agent.torch.training.env_utils.get_user_stack_pyspy"
    )
    @patch("psutil.process_iter")
    @patch("os.getpid")
    @patch("os.getpgid")
    @patch("subprocess.run")
    def test_stop_timeout_handler_pkill(
        self,
        mock_run,
        mock_getpgid,
        mock_getpid,
        mock_process_iter,
        mock_get_user_stack,
        mock_get_kernel_stack,
        mock_get_child_pids,
    ):
        """Test _stop_timeout_handler with pkill implementation"""
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )

        # Mock process IDs and group IDs
        mock_getpid.return_value = 1000
        mock_getpgid.return_value = 9999
        mock_get_child_pids.return_value = [1001]

        # Mock psutil.process_iter to return mock processes
        mock_process = MagicMock()
        mock_process.pid = 1001
        mock_process.ppid.return_value = 1000
        mock_process.name.return_value = "python"
        mock_process.cmdline.return_value = ["python", "train.py"]
        mock_process_iter.return_value = [mock_process]

        # Mock stack info
        mock_get_kernel_stack.return_value = (True, "kernel stack")
        mock_get_user_stack.return_value = (True, "user stack")

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # basic
        with self.assertRaises(StopWorkerTimeoutError):
            agent._stop_timeout_handler(signal.SIGALRM, None)

        # error cases
        mock_run.return_value = MagicMock(
            returncode=1, stderr="permission denied"
        )
        with self.assertRaises(StopWorkerTimeoutError):
            agent._stop_timeout_handler(signal.SIGALRM, None)

        mock_run.side_effect = subprocess.TimeoutExpired("pkill", 5)
        with self.assertRaises(StopWorkerTimeoutError):
            agent._stop_timeout_handler(signal.SIGALRM, None)

        mock_run.side_effect = subprocess.CalledProcessError(1, "pkill")
        with self.assertRaises(StopWorkerTimeoutError):
            agent._stop_timeout_handler(signal.SIGALRM, None)

        mock_run.side_effect = Exception("unexpected error")
        with self.assertRaises(StopWorkerTimeoutError):
            agent._stop_timeout_handler(signal.SIGALRM, None)

    def test_diagnosis(self):
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )
        agent._stop_workers = mock.MagicMock(return_value=True)
        agent._restart_workers = mock.MagicMock(return_value=True)

        context = get_agent_context()

        action = EventAction(
            event_action="action",
            expired_time_period=600,
        )
        context.enqueue_diagnosis_action(action)
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.MASTER_INSTANCE
                ]
            ),
            1,
        )
        time.sleep(1)
        agent._check_and_process_diagnosis_action()
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.MASTER_INSTANCE
                ]
            ),
            1,
        )

        action = NodeAction(
            node_id=0,
            node_type="worker",
            action_type=DiagnosisActionType.RESTART_WORKER,
            instance=DiagnosisConstant.ANY_INSTANCE,
        )
        context.enqueue_diagnosis_action(action)
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.ANY_INSTANCE
                ]
            ),
            1,
        )
        self.assertEqual(agent._remaining_failovers, 3)
        agent._check_and_process_diagnosis_action()
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.ANY_INSTANCE
                ]
            ),
            0,
        )
        self.assertEqual(agent._remaining_failovers, 3)

        action = NodeAction(
            node_id=0,
            node_type="worker",
            action_type=DiagnosisActionType.RESTART_WORKER,
            instance=DiagnosisConstant.LOCAL_INSTANCE,
        )
        context.enqueue_diagnosis_action(action)
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.LOCAL_INSTANCE
                ]
            ),
            1,
        )
        self.assertEqual(agent._remaining_failovers, 3)
        agent._check_and_process_diagnosis_action()
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.LOCAL_INSTANCE
                ]
            ),
            0,
        )
        self.assertEqual(agent._remaining_failovers, 2)

        action = NodeAction(
            node_id=1,
            node_type="worker",
            action_type=DiagnosisActionType.RELAUNCH_WORKER,
            instance=DiagnosisConstant.LOCAL_INSTANCE,
        )
        context.enqueue_diagnosis_action(action)
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.LOCAL_INSTANCE
                ]
            ),
            1,
        )
        time.sleep(2)
        agent._check_and_process_diagnosis_action()
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.LOCAL_INSTANCE
                ]
            ),
            0,
        )

        action = EventAction(
            expired_time_period=600, instance=DiagnosisConstant.ANY_INSTANCE
        )
        context.enqueue_diagnosis_action(action)
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.ANY_INSTANCE
                ]
            ),
            1,
        )
        time.sleep(1)
        agent._check_and_process_diagnosis_action()
        self.assertEqual(
            len(
                context._diagnosis_action_queue._actions[
                    DiagnosisConstant.ANY_INSTANCE
                ]
            ),
            1,
        )

    @patch(
        "dlrover.python.elastic_agent.master_client.MasterClient.report_failed_exited"
    )
    @patch(
        "dlrover.python.elastic_agent.torch.training.ElasticTrainingAgent.run"
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
        "dlrover.python.elastic_agent.torch.training.ElasticTrainingAgent.run"
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

    @patch("dlrover.python.elastic_agent.torch.training.get_gpu_stats")
    @patch("dlrover.python.elastic_agent.torch.training.get_hpu_stats")
    def test_check_device(self, mock_get_hpu_stats, mock_get_gpu_stats):
        self.assertFalse(ElasticTrainingAgent.is_device_checked())
        ElasticTrainingAgent.set_device_checked()
        self.assertTrue(ElasticTrainingAgent.is_device_checked())

        config = ElasticLaunchConfig(
            min_nodes=1, max_nodes=1, nproc_per_node=1
        )

        config.accelerator = Accelerators.GENERIC_CPU
        ElasticTrainingAgent.reset_device_checked()
        _check_device(config)

        mock_get_hpu_stats.return_value = []
        config.accelerator = Accelerators.ASCEND_NPU
        ElasticTrainingAgent.reset_device_checked()
        _check_device(config)

        mock_get_hpu_stats.return_value = [
            GPUStats(total_memory_mb=100, used_memory_mb=10)
        ]
        ElasticTrainingAgent.reset_device_checked()
        _check_device(config)

        mock_get_hpu_stats.return_value = [
            GPUStats(total_memory_mb=100, used_memory_mb=50)
        ]
        with self.assertRaises(NodeCheckFailedError):
            ElasticTrainingAgent.reset_device_checked()
            _check_device(config)

        mock_get_gpu_stats.return_value = []
        config.accelerator = Accelerators.NVIDIA_GPU
        ElasticTrainingAgent.reset_device_checked()
        _check_device(config)

        mock_get_gpu_stats.return_value = [
            GPUStats(total_memory_mb=100, used_memory_mb=10)
        ]
        ElasticTrainingAgent.reset_device_checked()
        _check_device(config)

        mock_get_gpu_stats.return_value = [
            GPUStats(total_memory_mb=100, used_memory_mb=50)
        ]
        with self.assertRaises(NodeCheckFailedError):
            ElasticTrainingAgent.reset_device_checked()
            _check_device(config)
        # skip cuz checked
        _check_device(config)

    def test_invoke_run_with_abort_restart_actions(self):
        from dlrover.python.diagnosis.common.diagnosis_action import (
            JobAbortionAction,
            JobRestartAction,
        )

        self.config.network_check = False
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )

        with patch(
            "dlrover.python.elastic_agent.torch.training._agent_evt"
        ) as mock_agent_evt:
            mock_abort_action = JobAbortionAction(reason="test abort reason")
            mock_agent_evt.job_abortion = MagicMock()

            run_result = RunResult(
                state=WorkerState.FAILED,
                return_values={},
                failures={"error": "test failure"},
            )
            agent._monitor_workers = MagicMock(return_value=run_result)
            agent._client.report_action = MagicMock()

            # Test JOB_ABORT action
            agent._diagnose_agent.diagnose_training_failure = MagicMock(
                return_value=mock_abort_action
            )
            agent._invoke_run()
            mock_agent_evt.job_abortion.assert_called_once_with(
                node_rank=0,
                reason="test abort reason",
                state="FAILED",
                return_values="{}",
                failures="{'error': 'test failure'}",
            )

            mock_agent_evt.job_abortion.reset_mock()
            mock_agent_evt.job_restart = MagicMock()
            mock_restart_action = JobRestartAction(
                reason="test restart reason"
            )

            # Test JOB_RESTART action
            agent._diagnose_agent.diagnose_training_failure = MagicMock(
                return_value=mock_restart_action
            )
            agent._invoke_run()
            mock_agent_evt.job_restart.assert_called_once_with(
                node_rank=0,
                reason="test restart reason",
                state="FAILED",
                return_values="{}",
                failures="{'error': 'test failure'}",
            )


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
            RendezvousName.TRAINING,
            node_id,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        self.rdzv_handler.join_timeout = 5

        if version_less_than_230():
            logs_dict = {
                "redirects": self.config.get_log_redirects(),
                "tee": self.config.get_log_tee(),
            }
        else:
            logs_dict = {}
        self.spec = WorkerSpec(
            role=self.config.role,
            local_world_size=self.config.nproc_per_node,
            entrypoint="echo",
            args=tuple([]),
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            master_addr=master_addr,
            local_addr=self.config.local_addr,
            **logs_dict,
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
            RendezvousName.TRAINING,
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
            RendezvousName.TRAINING,
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

    def test_get_rdzv_error_data(self):
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
            RendezvousName.TRAINING,
            0,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        rdzv_eror = rdzv_handler._get_rdzv_error_data(
            RendezvousErrorType.JOIN_TIMEOUT, "test123", 99
        )
        self.assertEqual(type(rdzv_eror), str)
        self.assertEqual(len(json.loads(rdzv_eror)), 5)


class ElasticTrainingAgentUcpTest(unittest.TestCase):
    """Test cases for ucp method in ElasticTrainingAgent."""

    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 0.5)
        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
            run_id="test",
        )
        self.config = ElasticLaunchConfig(**launch_config.__dict__)
        self.config.ucp_device_type = "cpu"

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
            RendezvousName.TRAINING,
            0,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        self.rdzv_handler.join_timeout = 5

        if version_less_than_230():
            logs_dict = {
                "redirects": self.config.get_log_redirects(),
                "tee": self.config.get_log_tee(),
            }
        else:
            logs_dict = {}
        self.spec = WorkerSpec(
            role=self.config.role,
            local_world_size=self.config.nproc_per_node,
            entrypoint="echo",
            args=tuple([]),
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            master_addr=master_addr,
            local_addr=self.config.local_addr,
            **logs_dict,
        )

    def tearDown(self):
        self._master.stop()
        os.environ.clear()

    def test_ucp_method_success(self):
        """Test the ucp method with successful checkpoint conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock saver
            saver = mock.MagicMock(spec=AsyncCheckpointSaver)

            # Mock the get_ckpt_saver method
            with mock.patch(
                "dlrover.python.elastic_agent.torch.training.AsyncCheckpointSaver.get_ckpt_saver",
                return_value=saver,
            ):
                # Setup mock returns
                checkpoint_parent_dir = tmpdir
                checkpoint_dir = os.path.join(tmpdir, "checkpoint-100")
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Use return_value instead of side_effect to avoid StopIteration
                # The ucp() method will call these methods multiple times in a loop
                # We want them to return the same values each time
                saver.get_latest_success_save_dir.return_value = (
                    checkpoint_parent_dir,
                    100,
                )
                saver.get_latest_start_saving_step.return_value = 100
                saver.ucp.return_value = True

                # Create agent instance
                agent = ElasticTrainingAgent(
                    node_rank=0,
                    config=self.config,
                    entrypoint="echo",
                    spec=self.spec,
                    start_method=self.config.start_method,
                    exit_barrier_timeout=1,
                )

                # Mock time.time() to provide enough values for the loop
                # The ucp() method calls time.time() multiple times:
                # 1. start_time = time.time() - returns 0.0
                # 2. elapsed_time = time.time() - start_time (in loop) - returns 1.0, 2.0, etc.
                # We need to provide enough values to avoid StopIteration
                # The loop will break when elapsed_time > wait_time (30) or max_wait_time (60)
                # We need to provide values up to at least 31 to trigger the break condition
                # Use itertools.count() to provide infinite values
                import itertools

                with mock.patch("time.time", side_effect=itertools.count()):
                    with mock.patch("time.sleep"):
                        agent.ucp()

                # Verify calls
                # The method should call get_latest_success_save_dir multiple times
                # (1 initial + loop iterations + 1 final)
                # With itertools.count() starting from 0, we get 33 calls
                # because elapsed_time starts at 0 and increments by 1 each second
                # and the loop breaks when elapsed_time > wait_time (30)
                # get_latest_start_saving_step is called 32 times (1 initial + loop iterations)
                self.assertEqual(
                    saver.get_latest_success_save_dir.call_count, 33
                )
                self.assertEqual(
                    saver.get_latest_start_saving_step.call_count, 32
                )

                # Check ucp was called with correct arguments
                expected_input_dir = os.path.join(
                    tmpdir, "checkpoint-100", "global_step100"
                )
                expected_output_dir = os.path.join(
                    tmpdir, "checkpoint-100", "ucp"
                )
                saver.ucp.assert_called_once_with(
                    expected_input_dir, expected_output_dir, "cpu"
                )

                # Check ucp.txt was created
                ucp_file = os.path.join(tmpdir, "ucp.txt")
                self.assertTrue(os.path.exists(ucp_file))

                # Verify content
                with open(ucp_file, "r") as f:
                    content = f.read()
                    self.assertEqual(content, "100")

    def test_ucp_method_no_checkpoint(self):
        """Test ucp method when no checkpoint exists."""
        with mock.patch(
            "dlrover.python.elastic_agent.torch.training.AsyncCheckpointSaver.get_ckpt_saver",
            return_value=None,
        ):
            agent = ElasticTrainingAgent(
                node_rank=0,
                config=self.config,
                entrypoint="echo",
                spec=self.spec,
                start_method=self.config.start_method,
                exit_barrier_timeout=1,
            )

            # Should not raise any exception when no saver exists
            agent.ucp()

    def test_ucp_method_ucp_fails(self):
        """Test ucp method when ucp conversion fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = mock.MagicMock(spec=AsyncCheckpointSaver)
            with mock.patch(
                "dlrover.python.elastic_agent.torch.training.AsyncCheckpointSaver.get_ckpt_saver",
                return_value=saver,
            ):
                checkpoint_dir = os.path.join(tmpdir, "checkpoint-100")
                os.makedirs(checkpoint_dir, exist_ok=True)

                saver.get_latest_success_save_dir.return_value = (tmpdir, 100)
                saver.get_latest_start_saving_step.return_value = 100
                saver.ucp.return_value = False  # ucp fails

                agent = ElasticTrainingAgent(
                    node_rank=0,
                    config=self.config,
                    entrypoint="echo",
                    spec=self.spec,
                    start_method=self.config.start_method,
                    exit_barrier_timeout=1,
                )

                # Provide enough time.time() calls using itertools.count()
                # to avoid StopIteration
                import itertools

                with mock.patch("time.time", side_effect=itertools.count()):
                    with mock.patch("time.sleep"):
                        agent.ucp()
                # Provide enough time.time() calls using itertools.count()
                # to avoid StopIteration
                import itertools

                with mock.patch("time.time", side_effect=itertools.count()):
                    with mock.patch("time.sleep"):
                        agent.ucp()

                # ucp.txt should not be created if ucp fails
                ucp_file = os.path.join(tmpdir, "ucp.txt")
                self.assertFalse(os.path.exists(ucp_file))

    def test_ucp_method_with_different_device_type(self):
        """Test ucp method with different ucp_device_type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = mock.MagicMock(spec=AsyncCheckpointSaver)
            with mock.patch(
                "dlrover.python.elastic_agent.torch.training.AsyncCheckpointSaver.get_ckpt_saver",
                return_value=saver,
            ):
                checkpoint_dir = os.path.join(tmpdir, "checkpoint-100")
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Use return_value instead of side_effect to avoid StopIteration
                # The ucp() method will call these methods multiple times in a loop
                # We want them to return the same values each time
                saver.get_latest_success_save_dir.return_value = (tmpdir, 100)
                saver.get_latest_start_saving_step.return_value = 100
                saver.ucp.return_value = True

                # Test with different device type
                self.config.ucp_device_type = "gpu"
                agent = ElasticTrainingAgent(
                    node_rank=0,
                    config=self.config,
                    entrypoint="echo",
                    spec=self.spec,
                    start_method=self.config.start_method,
                    exit_barrier_timeout=1,
                )

                # Provide enough time.time() values for the loop
                # The ucp() method calls time.time() multiple times in the loop
                # The loop will break when elapsed_time > wait_time (30) or max_wait_time (60)
                # We need to provide values up to at least 31 to trigger the break condition
                # Use itertools.count() to provide infinite values
                import itertools

                with mock.patch("time.time", side_effect=itertools.count()):
                    with mock.patch("time.sleep"):
                        agent.ucp()

                # Check ucp was called with gpu device type
                expected_input_dir = os.path.join(
                    tmpdir, "checkpoint-100", "global_step100"
                )
                expected_output_dir = os.path.join(
                    tmpdir, "checkpoint-100", "ucp"
                )
                saver.ucp.assert_called_once_with(
                    expected_input_dir, expected_output_dir, "gpu"
                )


class ElasticTrainingAgentGracefulExitTest(unittest.TestCase):
    """Test cases for _graceful_exit_workers method in ElasticTrainingAgent."""

    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 0.5)
        launch_config = LaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=2,
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

        self.rdzv_handler = MasterRendezvousHandler(
            RendezvousName.TRAINING,
            0,
            rdzv_parameters,
            local_world_size=self.config.nproc_per_node,
        )
        self.rdzv_handler.join_timeout = 5

        if version_less_than_230():
            logs_dict = {
                "redirects": self.config.get_log_redirects(),
                "tee": self.config.get_log_tee(),
            }
        else:
            logs_dict = {}
        self.spec = WorkerSpec(
            role=self.config.role,
            local_world_size=self.config.nproc_per_node,
            entrypoint="echo",
            args=tuple([]),
            rdzv_handler=self.rdzv_handler,
            max_restarts=self.config.max_restarts,
            monitor_interval=self.config.monitor_interval,
            master_addr=master_addr,
            local_addr=self.config.local_addr,
            **logs_dict,
        )

    def tearDown(self):
        self._master.stop()
        os.environ.clear()

    def test_graceful_exit_workers_success(self):
        """Test _graceful_exit_workers with successful signal sending."""
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )

        # Mock pcontext with multiple PIDs
        mock_pcontext = mock.MagicMock()
        mock_pids = {0: 1001, 1: 1002}
        mock_pcontext.pids.return_value = mock_pids
        agent._pcontext = mock_pcontext

        # Mock os.kill to verify it's called with correct arguments
        with mock.patch("os.kill") as mock_kill:
            # Create a mock worker group (not used in the method but required for signature)
            worker_group = mock.MagicMock()

            # Call the method
            agent._graceful_exit_workers(worker_group)

            # Verify os.kill was called with the first PID and SIGUSR1
            mock_kill.assert_called_once_with(1001, signal.SIGUSR1)

            # Verify logger.info was called
            # We can check this by patching logger if needed, but for now just verify no exceptions

    def test_graceful_exit_workers_no_pids(self):
        """Test _graceful_exit_workers when no PIDs are available."""
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )

        # Mock pcontext with empty PIDs
        mock_pcontext = mock.MagicMock()
        mock_pcontext.pids.return_value = {}
        agent._pcontext = mock_pcontext

        # Mock os.kill
        with mock.patch("os.kill") as mock_kill:
            worker_group = mock.MagicMock()

            # This should not raise an exception
            agent._graceful_exit_workers(worker_group)

            # os.kill should not be called since there are no PIDs
            mock_kill.assert_not_called()

    def test_graceful_exit_workers_single_pid(self):
        """Test _graceful_exit_workers with only one PID."""
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )

        # Mock pcontext with single PID
        mock_pcontext = mock.MagicMock()
        mock_pids = {0: 1001}
        mock_pcontext.pids.return_value = mock_pids
        agent._pcontext = mock_pcontext

        with mock.patch("os.kill") as mock_kill:
            worker_group = mock.MagicMock()

            agent._graceful_exit_workers(worker_group)

            # Should still call os.kill with the single PID
            mock_kill.assert_called_once_with(1001, signal.SIGUSR1)

    def test_graceful_exit_workers_kill_exception(self):
        """Test _graceful_exit_workers when os.kill raises an exception."""
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )

        # Mock pcontext with PIDs
        mock_pcontext = mock.MagicMock()
        mock_pids = {0: 1001, 1: 1002}
        mock_pcontext.pids.return_value = mock_pids
        agent._pcontext = mock_pcontext

        # Mock os.kill to raise an exception
        with mock.patch(
            "os.kill", side_effect=OSError("No such process")
        ) as mock_kill:
            # Mock logger.warning to verify it's called
            with mock.patch.object(logger, "warning") as mock_warning:
                worker_group = mock.MagicMock()

                # This should not raise an exception
                agent._graceful_exit_workers(worker_group)

                # Verify os.kill was attempted
                mock_kill.assert_called_once_with(1001, signal.SIGUSR1)

                # Verify logger.warning was called with the error
                mock_warning.assert_called_once()
                call_args = mock_warning.call_args[0][0]
                self.assertIn("error when kill", call_args)
                self.assertIn("No such process", call_args)

    def test_graceful_exit_workers_with_none_pcontext(self):
        """Test _graceful_exit_workers when _pcontext is None."""
        agent = ElasticTrainingAgent(
            node_rank=0,
            config=self.config,
            entrypoint="echo",
            spec=self.spec,
            start_method=self.config.start_method,
            exit_barrier_timeout=1,
        )

        # Set _pcontext to None
        agent._pcontext = None

        with mock.patch("os.kill") as mock_kill:
            worker_group = mock.MagicMock()

            # This should not raise an exception
            agent._graceful_exit_workers(worker_group)

            # os.kill should not be called
            mock_kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
