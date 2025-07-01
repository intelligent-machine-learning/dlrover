import os
from typing import List

import ray.actor

from dlrover.python.common.constants import CommunicationType, NodeEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import RayMasterClient
from dlrover.python.unified.util.test_hooks import init_coverage
from dlrover.trainer.torch.elastic_run import main

init_coverage()  # support coverage for runner actor


class ElasticRunner:
    """The actual worker process that runs the training job

    Currently, it only runs the elastic agent, but in the future,
    agent features will be moved into workers, and runners will directly run training processes.
    """

    def __init__(self, master: str, args: List[str]) -> None:
        """Initialize the runner with master address and command line arguments."""
        self.master = master
        self.args = args

        os.environ[NodeEnv.DLROVER_MASTER_SERVICE_TYPE] = (
            CommunicationType.COMM_SERVICE_RAY
        )
        RayMasterClient.register_master_actor(master)

    def run(self):
        """Run the elastic agent with the provided command line arguments."""
        logger.info(
            f"Running elastic agent with master: {self.master}, "
            f"args: {self.args}"
        )
        main(self.args)

    def shutdown(self):
        """Shutdown the worker process."""
        ray.actor.exit_actor()