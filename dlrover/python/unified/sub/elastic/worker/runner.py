import os
from typing import List

import ray.actor

from dlrover.python.common.constants import CommunicationType, NodeEnv
from dlrover.python.common.log import default_logger as logger
from dlrover.python.elastic_agent.master_client import RayMasterClient
from dlrover.python.training_event.predefined._dlrover import DLRoverAgentEvent
from dlrover.python.unified.util.test_hooks import init_coverage
from dlrover.trainer.torch.elastic_run import (
    _elastic_config_from_args,
    launch_agent,
    parse_args,
    wait_pre_check,
)

init_coverage()  # support coverage for runner actor


class ElasticRunner:
    """The actual worker process that runs the training job

    Currently, it only runs the elastic agent, but in the future,
    agent features will be moved into workers, and runners will directly run training processes.
    """

    def __init__(self, job_name: str, master: str, args: List[str]) -> None:
        """Initialize the runner with master address and command line arguments."""
        self.job_name = job_name
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
        parsed = parse_args(self.args)
        evt = DLRoverAgentEvent.singleton_instance()
        evt.start(pid=vars(parsed))

        config, cmd, cmd_args = _elastic_config_from_args(parsed)
        config.run_id = self.job_name
        config.role = "dlrover-trainer"

        # TODO remove this, as pre-check move to elastic-worker
        wait_pre_check(config)

        launch_agent(config, cmd, list(cmd_args))

    def shutdown(self):
        """Shutdown the worker process."""
        ray.actor.exit_actor()
