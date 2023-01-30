import unittest
from dlrover.python.util.state.store_mananger import StoreManager
import os 


class StoreManagerTest(unittest.TestCase):
    def test_memory_store_mananger(self):
        os.environ["state_backend_type"] = "Memory"
        store_manager = StoreManager("jobname", "state", "config")
        self.assertEqual(store_manager.store_type(), None)
        memory_store_manager = store_manager.build_store_manager()
        self.assertEqual(memory_store_manager.store_type(), "Memory")
        memory_store = memory_store_manager.build_store()
        memory_store.put("key", "value")
        self.assertEqual(memory_store.get("key"), "value")
        memory_store.delete("key")
        self.assertEqual(memory_store.get("key"), None)
