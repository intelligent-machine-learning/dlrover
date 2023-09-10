import unittest
import os
import json
import time
from unittest.mock import MagicMock, patch
from dlrover.python.elastic_agent.config.paral_config_tuner import ParalConfigTuner
from dlrover.trainer.constants.torch import WorkerEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import GlobalMasterClient
MOCKED_CONFIG = {
    "dataloader": {
        "batch_size": 3,
        "dataloader_name": 2,
        "num_workers": 4,
        "pin_memory": 0,
        "version": 1,
    },
    "optimizer": {"learning_rate": 0.0, "optimizer_name": 6, "version": 5},
}


class TestParalConfigTuner(unittest.TestCase):
    def setUp(self):
        self.tuner = ParalConfigTuner()

    def test_set_paral_config(self):
        self.tuner._set_paral_config()
        self.assertTrue(os.path.exists(self.tuner.config_dir))
        self.assertEqual(
            os.environ[WorkerEnv.PARAL_CONFIG_PATH.name],
            WorkerEnv.PARAL_CONFIG_PATH.default,
        )

    def test_read_paral_config(self):
        with open(self.tuner.config_path, "w") as json_file:
            json.dump(MOCKED_CONFIG, json_file)
        config = self.tuner._read_paral_config(self.tuner.config_path)
        self.assertEqual(config, MOCKED_CONFIG)

    def test_read_paral_config_file_not_found(self):
        os.remove(self.tuner.config_path)
        config = self.tuner._read_paral_config(self.tuner.config_path)
        self.assertIsNone(config)


if __name__ == "__main__":
    unittest.main()
