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

import json
import os
import shutil
import socket
import tempfile
import time
import unittest
from unittest import mock
from unittest.mock import patch

from torch.distributed.elastic.agent.server.api import WorkerSpec, WorkerState
from torch.distributed.elastic.agent.server.local_elastic_agent import (
    LocalElasticAgent,
)
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.launcher.api import LaunchConfig

from dlrover.python.common.constants import (
    Accelerators,
    AscendConstants,
    ConfigPath,
    RendezvousName,
)
from dlrover.python.common.storage import PosixDiskStorage
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
    MasterRendezvousHandler,
    NodeCheckElasticAgent,
    RendezvousOutSyncError,
    _create_check_agent,
    _create_worker_spec,
    _get_local_ip,
    _set_paral_config,
    comm_perf_check,
    node_health_check,
)
from dlrover.python.tests.test_utils import start_local_master


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

    def tearDown(self):
        self._master.stop()

    def test_node_unit(self):
        node_unit = int(self.rdzv_handler._rdzv_params.get("node_unit", "1"))
        self.assertEqual(node_unit, 2)

    def test_auto_configure(self):
        config = ElasticLaunchConfig(
            min_nodes=1,
            max_nodes=1,
            nproc_per_node=8,
            run_id="test",
            auto_config=True,
        )
        os.environ["NODE_NUM"] = "4"
        config.auto_configure_params()
        self.assertEqual(config.max_nodes, 4)
        self.assertEqual(config.min_nodes, 4)
        self.assertTrue(config.network_check)

    def test_rank0_rendzevous(self):
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

    def test_rank1_rendzevous(self):
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
        store.set("MASTER_ADDR", "127.0.0.1".encode())
        store.set("MASTER_PORT", "12345".encode())

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

    def test_get_local_ip(self):
        local_ip = _get_local_ip()
        self.assertNotEqual(local_ip, "")
        os.environ["POD_IP"] = "127.0.0.1"
        local_ip = _get_local_ip()
        self.assertEqual(local_ip, "127.0.0.1")

    def test_initialize_worker(self):
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
        with self.assertRaises(TimeoutError):
            agent._initialize_workers(agent._worker_group)
            agent._save_ckpt_future


class ElasticTrainingAgentRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 1)
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

    def tearDown(self):
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

    def test_report_resource_with_step(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            config_file = os.path.join(tmpdirname, "runtime_metrics.json")
            monitor = TorchTrainingMonitor(config_file)
            monitor.start()
            monitor.report_resource_with_step()
            self.assertEqual(self._master.speed_monitor._global_step, 0)
            record = {"step": 100, "timestamp": time.time()}
            with open(config_file, "w") as f:
                f.write(json.dumps(record))

            monitor.report_resource_with_step()
            self.assertEqual(self._master.speed_monitor._global_step, 100)

    def test_check_network_rdzv_for_elastic_training(self):
        self._master.rdzv_managers[
            RendezvousName.NETWORK_CHECK
        ].join_rendezvous(0, 0, 8)
        with self.assertRaises(RendezvousOutSyncError):
            self.rdzv_handler._check_network_rdzv_for_elastic_training()

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
        agent._stop_workers(None, 3)

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
                agent._stop_workers(None, 3)
                self.fail()
            except TimeoutError:
                self.assertTrue(True)


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

    def tearDown(self):
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


class MasterRendezvousHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._master, addr = start_local_master()
        MasterClient._instance = build_master_client(addr, 0.5)

    def tearDown(self):
        self._master.stop()

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
        with self.assertRaises(TimeoutError):
            rdzv_handler.next_rendezvous()


if __name__ == "__main__":
    unittest.main()
