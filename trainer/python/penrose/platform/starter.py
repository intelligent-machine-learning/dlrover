from penrose.util.log_util import default_logger as logger
import os
from penrose.mock.tf_process_scheduler import TFProcessScheduler
from penrose.util.args_util import get_parsed_args
from penrose.constants.platform_constants import PlatformConstants
import penrose
from penrose.worker.tf_kubernetes_worker import TFKubernetesWorker

def print_info(append_detail=False):
    """Print penrose information"""
    penrose_dir = os.path.dirname(penrose.__file__)
    file_path = os.path.join(penrose_dir, "COMMIT_INFO")
    if not os.path.exists(file_path):
        logger.info("Whl is not built by sh build.sh, please be careful.")
        return
    with open(file_path, encoding="utf-8") as fd:
        commit_id = fd.readline().strip()
        user = fd.readline().strip()
        time = fd.readline().strip()
    logger.info(penrose.logo_string)
    logger.info("-" * 30)
    logger.info("Penrose version: %s", penrose.__version__)
    logger.info("Build by: %s", user)
    logger.info("Build time: %s", time)
    logger.info("Commit id: %s", commit_id)
    logger.info("Pid: %s", os.getpid())
    logger.info("CWD: %s", os.getcwd())
    logger.info("-" * 30)


def execute(args):
    """run routine"""
    platform = args.platform.upper()
    if platform == PlatformConstants.Kubernetes():
        worker = TFKubernetesWorker(args)
    elif platform in [PlatformConstants.Local()]:
        # local mode, actually we use a scheduler
        logger.info(
            "create ProcessScheduler with run_type = ProcessScheduler"
        )
        worker = TFProcessScheduler(
            ps_num=args.ps_num,
            worker_num=args.worker_num,
            evaluator_num=args.evaluator_num,
            conf=args.conf,
            parsed_args=args,
        )

    logger.info(
        "Running platform: %s, worker action: %s",
        args.platform,
        args.worker_action,
    )
    if args.worker_action == PlatformConstants.WorkerActionRun():
        return worker.run()


def run(other_arguments=None):
    """Entrance
    Args:
        other_arguments: dict of complex arguments
    """
    args = get_parsed_args()
    return execute(args)