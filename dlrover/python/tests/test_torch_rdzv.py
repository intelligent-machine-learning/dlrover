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

import datetime
import pickle
import unittest

from torch.distributed.elastic.rendezvous.dynamic_rendezvous import (
    _RendezvousState,
)

from dlrover.python.common.constants import NodeType
from dlrover.python.common.node import Node
from dlrover.python.elastic_agent.torch.master_kv_store import MasterKVStore
from dlrover.python.elastic_agent.torch.rdzv_backend import (
    DlroverRendezvousBackend,
)
from dlrover.python.master.elastic_training.rdzv_service import (
    TorchRendezvousService,
)


class MasterKVStoreTest(unittest.TestCase):
    def test_kv_store_api(self):
        kv_store = MasterKVStore("dlrover/torch/test")
        key = "key0"
        kv_store.set(key, "1")
        value = kv_store.get(key)
        self.assertEqual(int(value), 1)
        kv_store.add(key, 2)
        value = kv_store.get(key)
        self.assertEqual(int(value), 3)
        kv_store.wait([key])
        try:
            kv_store.wait(
                ["aa"], override_timeout=datetime.timedelta(seconds=0.01)
            )
        except Exception as e:
            self.assertIsInstance(e, LookupError)


class DlroverRendezvousBackendTest(unittest.TestCase):
    def test_set_and_get_state(self):
        rdzv_backend = DlroverRendezvousBackend(
            "test-rdzv",
            "torch.elastic.redzv.",
        )
        state_bits = pickle.dumps(_RendezvousState())
        new_state, token, success = rdzv_backend.set_state(state_bits, 1)
        self.assertEqual(new_state, state_bits)
        self.assertTrue(success)
        new_state, token, success = rdzv_backend.set_state(state_bits, 1)
        self.assertFalse(success)
        new_state, token = rdzv_backend.get_state()
        self.assertEqual(new_state, state_bits)
        self.assertEqual(token, 1)


class RdzvServiceTest(unittest.TestCase):
    def test_rdzv_service(self):
        rdzv_svc = TorchRendezvousService()
        rdzv_key = "test-rdzv-0"
        state = pickle.dumps("aaaaa")
        state_bits = pickle.dumps(state)
        participants = {"worker-0": 0}
        wait_list = ["worker-1"]
        host = "worker-0"
        rdzv_svc._participants = ["worker-0", "worker-1"]
        rdzv_svc.set_state(
            rdzv_key, state_bits, 1, participants, wait_list, host
        )
        new_state_bits, token = rdzv_svc.get_state(host, rdzv_key)
        self.assertEqual(new_state_bits, state_bits)
        self.assertEqual(token, 0)

    def test_scale_down_worker_base2(self):
        rdzv_svc = TorchRendezvousService()
        worker0 = Node(NodeType.WORKER, 0, name="worker-0")
        rdzv_svc.add_alive_worker(worker0)
        self.assertListEqual(rdzv_svc._participants, ["worker-0"])
        rdzv_svc._alive_workers = ["worker-0", "worker-1", "worker-2"]
        worker2 = Node(NodeType.WORKER, 0, name="worker-2")
        rdzv_svc.remove_alive_worker(worker2)
        rdzv_svc._scale_down_ts -= 400
        rdzv_svc._scale_down_worker_base2()
        self.assertEqual(rdzv_svc._participants, ["worker-0", "worker-1"])
