import os
from argparse import REMAINDER, ArgumentParser

from distutils.util import strtobool
from torch.distributed.argparse_util import check_env, env
from torch.distributed.run import run as elastic_run

from atorch.distributed.hooks import hook_set_master_addr_port
from atorch.distributed.launch import main as normal_run
from atorch.distributed.launch import parse_args as parse_static_args
from atorch.fault_tolerance.api import run as fault_tolerant_run


def parse_elastic_args(args=None, mode="elastic"):
    """Helper function parsing the command line options when using elastic training."""
    parser = ArgumentParser(description="ATorch Distributed Elastic Training Launcher")

    #
    # Worker/node size related arguments.
    #
    parser.add_argument(
        "--nnodes",
        type=str,
        default=None,
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    )
    parser.add_argument(
        "--nproc_per_node",
        action=env,
        type=str,
        default="1",
        help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
    )

    #
    # Rendezvous related arguments
    #

    parser.add_argument(
        "--rdzv_backend",
        type=str,
        default=None,
        help="Rendezvous backend.",
    )
    parser.add_argument(
        "--rdzv_endpoint",
        type=str,
        default=None,
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv_id",
        type=str,
        default=None,
        help="User-defined group id.",
    )
    parser.add_argument(
        "--rdzv_conf",
        action=env,
        type=str,
        default="",
        help="Additional rendezvous configuration (<key1>=<value1>,<key2>=<value2>,...).",
    )
    parser.add_argument(
        "--standalone",
        action=check_env,
        help="Start a local standalone rendezvous backend that is represented by a C10d TCP store "
        "on port 29400. Useful when launching single-node, multi-worker job. If specified "
        "--rdzv_backend, --rdzv_endpoint, --rdzv_id are auto-assigned; any explicitly set values "
        "are ignored.",
    )

    #
    # User-code launch related arguments.
    #

    parser.add_argument(
        "--max_restarts",
        action=env,
        type=int,
        default=100,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--monitor_interval",
        action=env,
        type=float,
        default=5,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--start_method",
        action=env,
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method to use when creating workers.",
    )
    parser.add_argument(
        "--role",
        action=env,
        type=str,
        default="default",
        help="User-defined role for the workers.",
    )
    parser.add_argument(
        "-m",
        "--module",
        action=check_env,
        help="Change each process to interpret the launch script as a Python module, executing "
        "with the same behavior as 'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        action=check_env,
        help="Skip prepending the training script with 'python' - just execute it directly. Useful "
        "when the script is not a Python script.",
    )

    parser.add_argument(
        "--run_path",
        action=check_env,
        help="Run the training script with runpy.run_path in the same interpreter."
        " Script must be provided as an abs path (e.g. /abs/path/script.py)."
        " Takes precedence over --no_python.",
    )
    parser.add_argument(
        "--log_dir",
        action=env,
        type=str,
        default=None,
        help="Base directory to use for log files (e.g. /var/log/torch/elastic). The same "
        "directory is re-used for multiple runs (a unique job-level sub-directory is created with "
        "rdzv_id as the prefix).",
    )
    parser.add_argument(
        "-r",
        "--redirects",
        action=env,
        type=str,
        default="0",
        help="Redirect std streams into a log file in the log directory (e.g. [-r 3] redirects "
        "both stdout+stderr for all workers, [-r 0:1,1:2] redirects stdout for local rank 0 and "
        "stderr for local rank 1).",
    )
    parser.add_argument(
        "-t",
        "--tee",
        action=env,
        type=str,
        default="0",
        help="Tee std streams into a log file and also to console (see --redirects for format).",
    )

    #
    # Positional arguments.
    #

    parser.add_argument(
        "training_script",
        type=str,
        help="Full path to the (single GPU) training program/script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )

    # Rest from the training program.
    parser.add_argument("training_script_args", nargs=REMAINDER)

    args = parser.parse_args(args)
    args = get_rendezvous_info(args, mode)
    return args


def get_rendezvous_info(args, mode):
    """Set rendezvous related arguments from env"""
    assert mode in ("elastic", "fault_tolerant")
    edl_enabled = strtobool(os.getenv("ELASTICDL_ENABLED", "false"))
    if args.nnodes is None:
        if mode == "elastic":
            if edl_enabled:
                # env HIGH_PRIORITY_NODES and HIGH_PRIORITY_NODES are both set by edl
                high_nnodes = int(os.getenv("HIGH_PRIORITY_NODES", "0"))
                low_nnodes = int(os.getenv("LOW_PRIORITY_NODES", "1"))
                if high_nnodes == 0 and low_nnodes == 0:
                    min_size = max_size = 1
                elif high_nnodes == 0 or low_nnodes == 0:
                    min_size = max_size = high_nnodes + low_nnodes
                else:
                    min_size = high_nnodes
                    max_size = high_nnodes + low_nnodes
                args.nnodes = f"{min_size}:{max_size}"
            else:
                raise ValueError("If you want to use elastic training, `nnodes` should not be None.")
        else:  # fault_tolerant
            if edl_enabled:
                # env WORKER_NUM is set by edl
                workers_num = os.getenv("WORKER_NUM", "1")
            else:
                workers_num = os.getenv("WORLD_SIZE", "1")
            args.nnodes = workers_num
    if args.rdzv_backend is None:
        args.rdzv_backend = "c10d"
    if args.rdzv_endpoint is None:
        if edl_enabled:
            # env RDZV_ENDPOINT is set by edl
            rdzv_endpoint = os.getenv("RDZV_ENDPOINT", None)
            if rdzv_endpoint is not None:
                port = "29400"  # pytorch default port
                args.rdzv_endpoint = f"{rdzv_endpoint}:{port}"
            else:
                args.rdzv_endpoint = "localhost:29400"
        else:
            master_addr = os.getenv("MASTER_ADDR", "localhost")
            master_port = os.getenv("MASTER_PORT", "29400")
            args.rdzv_endpoint = f"{master_addr}:{master_port}"
    if args.rdzv_id is None:
        rdzv_id = os.getenv("APP_ID", None)
        if rdzv_id is None:
            rdzv_id = os.getenv("AISTUDIO_JOB_NAME", f"atorch-{mode}-job")
        args.rdzv_id = rdzv_id
    if not args.standalone:
        if args.nnodes == "1":
            args.standalone = True
    return args


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--relaunch_on_hanging",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--elastic",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fault_tolerant",
        default=False,
        action="store_true",
    )
    args, unknown_args = parser.parse_known_args()
    if args.elastic is True:
        args = parse_elastic_args(unknown_args, "elastic")
        hook_set_master_addr_port()
        elastic_run(args)
    elif args.relaunch_on_hanging is True or args.fault_tolerant is True:
        args = parse_elastic_args(unknown_args, "fault_tolerant")
        fault_tolerant_run(args)
    else:
        args = parse_static_args(unknown_args)
        normal_run(args)


if __name__ == "__main__":
    main()
