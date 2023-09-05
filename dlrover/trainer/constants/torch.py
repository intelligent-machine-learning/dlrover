from dlrover.trainer.constants.constants import Constant


class WorkerEnv(Constant):
    PARAL_CONFIG_PATH = Constant(
        "DLROVER_PARAL_CONFIG_PATH",
        "/tmp/dlrover/auto_paral_config.json",
    )
