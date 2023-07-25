import os
from argparse import REMAINDER, ArgumentParser

from distutils.util import strtobool
from torch.distributed.argparse_util import check_env, env
from torch.distributed.run import run as torchrun

from atorch.common.log_utils import default_logger as logger
from atorch.distributed.hooks import hook_set_master_addr_port
from atorch.distributed.launch import main as normal_run
from atorch.distributed.launch import parse_args as parse_static_args
from atorch.fault_tolerance.api import run as fault_tolerant_run

try:
    from dlrover.trainer.torch.elastic_run import run as dlrover_run
except ImportError:
    logger.warning("DLRrover is not installed and torchrun is used.")
    dlrover_run = torchrun


def parse_fault_tolerant_or_elastic_args(args=None, mode="elastic"):
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
        "--nproc-per-node",
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
        "--rdzv-backend",
        type=str,
        default=None,
        help="Rendezvous backend.",
    )
    parser.add_argument(
        "--rdzv_endpoint",
        "--rdzv-endpoint",
        type=str,
        default=None,
        help="Rendezvous backend endpoint; usually in form <host>:<port>.",
    )
    parser.add_argument(
        "--rdzv_id",
        "--rdzv-id",
        type=str,
        default=None,
        help="User-defined group id.",
    )
    parser.add_argument(
        "--rdzv_conf",
        "--rdzv-conf",
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
    parser.add_argument(
        "--network_check",
        "--network-check",
        action=check_env,
        help="Whether to check network before starting training process.",
    )
    parser.add_argument(
        "--node_unit",
        "--node-unit",
        type=int,
        action=env,
        default=1,
        help="The number unit of nodes to schedule. The number of "
        "nodes should be a multiple of node_unit. The argument is only"
        "valid when dlrover-run is used.",
    )

    #
    # User-code launch related arguments.
    #

    parser.add_argument(
        "--max_restarts",
        "--max-restarts",
        action=env,
        type=int,
        default=100,
        help="Maximum number of worker group restarts before failing.",
    )
    parser.add_argument(
        "--monitor_interval",
        "--monitor-interval",
        action=env,
        type=float,
        default=5,
        help="Interval, in seconds, to monitor the state of workers.",
    )
    parser.add_argument(
        "--start_method",
        "--start-method",
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
        "--no-python",
        action=check_env,
        help="Skip prepending the training script with 'python' - just execute it directly. Useful "
        "when the script is not a Python script.",
    )

    parser.add_argument(
        "--run_path",
        "--run-path",
        action=check_env,
        help="Run the training script with runpy.run_path in the same interpreter."
        " Script must be provided as an abs path (e.g. /abs/path/script.py)."
        " Takes precedence over --no_python.",
    )
    parser.add_argument(
        "--log_dir",
        "--log-dir",
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
    # Backwards compatible parameters with caffe2.distributed.launch.
    #

    parser.add_argument(
        "--node-rank",
        "--node_rank",
        type=int,
        action=env,
        default=0,
        help="Rank of the node for multi-node distributed training.",
    )
    parser.add_argument(
        "--master-addr",
        "--master_addr",
        default="127.0.0.1",
        type=str,
        action=env,
        help="Address of the master node (rank 0) that only used for static rendezvous. It should "
        "be either the IP address or the hostname of rank 0. For single node multi-proc training "
        "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
        "`[0:0:0:0:0:0:0:1]`.",
    )
    parser.add_argument(
        "--master-port",
        "--master_port",
        default=29500,
        type=int,
        action=env,
        help="Port on the master node (rank 0) to be used for communication during distributed "
        "training. It is only used for static rendezvous.",
    )
    parser.add_argument(
        "--local-addr",
        "--local_addr",
        default=None,
        type=str,
        action=env,
        help="Address of the local node. If specified, will use the given address for connection. "
        "Else, will look up the local node address instead. Else, it will be default to local "
        "machine's FQDN.",
    )

    parser.add_argument(
        "--rank_log_dir",
        "--rank-log-dir",
        type=str,
        default="",
        help="Log dir for each rank, format is path,mode. path is dir to dump log"
        "(if path is not exists, create it).mode is wb or ab, wb is overwrite, ab"
        "is append",
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
        "--fault-tolerant",
        default=False,
        action="store_true",
    )
    parser.add_argument("--ib_stat_kw", type=str, default="", help="interval,count,path")

    args, unknown_args = parser.parse_known_args()
    if [args.relaunch_on_hanging, args.elastic, args.fault_tolerant].count(True) > 1:
        raise ValueError(
            "Among the three values of relaunch_on_hanging, fault_tolerant, and elastic, only one can be "
            f"selected. But got relaunch_on_hanging={args.relaunch_on_hanging}, fault_tolerant={args.fault_tolerant},"
            f" elastic={args.elastic}."
        )
    if args.ib_stat_kw:
        split = args.ib_stat_kw.split(",")
        if len(split) != 3:
            raise ValueError(f"ib_stat_kw error, your input is {args.ib_stat_kw}")
        interval, count, path = split
        interval = int(interval)
        count = int(count)
        from atorch.utils.ib_monitor import IBStat

        IBStat(interval, count, path)
    if args.elastic is True:
        args = parse_fault_tolerant_or_elastic_args(unknown_args, "elastic")
        hook_set_master_addr_port()
        elastic_run(args)
    elif args.fault_tolerant is True:
        args = parse_fault_tolerant_or_elastic_args(unknown_args, "fault_tolerant")
        hook_set_master_addr_port()
        elastic_run(args)
    elif args.relaunch_on_hanging is True:
        args = parse_fault_tolerant_or_elastic_args(unknown_args, "fault_tolerant")
        fault_tolerant_run(args)
    else:
        args = parse_static_args(unknown_args)
        normal_run(args)


def elastic_run(args):
    if os.getenv("DLROVER_MASTER_ADDR", ""):
        dlrover_run(args)
    else:
        torchrun(args)


if __name__ == "__main__":
    main()
