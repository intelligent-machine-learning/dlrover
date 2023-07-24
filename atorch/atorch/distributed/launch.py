r"""
This is a revised version of pytorch 1.7 distributed.launch from
github.com/pytorch/pytorch/blob/1.7/torch/distributed/launch.py
Changes:
  0. launcher uses env to pass MASTER_ADDR/MASTER_PORT/WORLD_SIZE/LOCAL_SIZE
     to training script.
  1. pytorch operator will set MASTER_ADDR/MASTER_PORT/WORLD_SIZE env.
     Thus, env values are used if corresponding args are not set.
     WORLD_SIZE is revised to env(WORLD_SIZE) * nproc_per_node.
  2. add not_use_env set default to True to use env for LOCAL_RANK.
  3. add a logic to check subprocess status and exit if any fails.
  4. support to launch a process with nsys for profiling.
  5. non-master nodes wait master before launch.

When submitting a pytorch job, set the command as:
::

    >>> python -m atorch.distributed.launch --nproc_per_node=NPROC_PER_NODE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

NPROC_PER_NODE usually is the number of gpus in a node for DDP.

--------------------------------------------------------------------------

`torch.distributed.launch` is a module that spawns up multiple distributed
training processes on each of the training nodes.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be benefitial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc_per_node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

**How to use this module:**

1. Single-Node multi-process distributed training

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

2. Multi-Node multi-process distributed training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

Node 2:

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

3. To look up what optional arguments this module offers:

::

    >>> python -m torch.distributed.launch --help


**Important Notices:**

1. This utility and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

::

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local_rank", type=int)
    >>> args = parser.parse_args()

Set your device to local rank using either

::

    >>> torch.cuda.set_device(arg.local_rank)  # before your code runs

or

::

    >>> with torch.cuda.device(arg.local_rank):
    >>>    # your code to run

3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses ``env://``, which is the only supported ``init_method``
by this module.

::

    torch.distributed.init_process_group(backend='YOUR BACKEND',
                                         init_method='env://')

4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.

::

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[arg.local_rank],
                                                      output_device=arg.local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility

5. Another way to pass ``local_rank`` to the subprocesses via environment
variable ``LOCAL_RANK``. This behavior is enabled when you launch the script
with ``--use_env=True``. You must adjust the subprocess example above to
replace ``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local_rank`` when you specify this flag.

.. warning::

    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.

"""


import atexit
import os
import socket
import subprocess
import sys
import time
from argparse import REMAINDER, ArgumentParser
from datetime import timedelta

import torch.distributed as dist

from atorch.common.util_func import find_free_port, get_ip_address


def parse_args(args=None):
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="PyTorch distributed training launch "
        "helper utility that will spawn up "
        "multiple distributed processes"
    )

    # Optional arguments for nsys profiling
    parser.add_argument(
        "--nsys_profiling_rank",
        type=int,
        default=-1,
        help="If >= 0, launch the corresponding process with nsys",
    )
    parser.add_argument(
        "--nsys_command",
        type=str,
        default="/usr/local/cuda/bin/nsys",
        help="nsys command path",
    )

    # Optional arguments for the launch helper
    parser.add_argument(
        "--nnodes",
        type=int,
        default=0,
        help="The number of nodes to use for distributed " "training",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=-1,
        help="The rank of the node for multi-node distributed " "training",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="The number of processes to launch on each node, "
        "for GPU training, this is recommended to be set "
        "to the number of GPUs in your system so that "
        "each process can be bound to a single GPU.",
    )
    parser.add_argument(
        "--master_addr",
        default="",
        type=str,
        help="Master node (rank 0)'s address, should be either "
        "the IP address or the hostname of node 0, for "
        "single node multi-proc training, the "
        "--master_addr can simply be 127.0.0.1",
    )
    parser.add_argument(
        "--master_port",
        default=0,
        type=int,
        help="Master node (rank 0)'s free port that needs to "
        "be used for communication during distributed "
        "training",
    )
    parser.add_argument(
        "--master_port2",
        default=0,
        type=int,
        help="Master node (rank 0)'s second free port that "
        "needs to be used for communication during data "
        "reading and preprocessing",
    )
    parser.add_argument(
        "--not_use_env",
        default=False,
        action="store_true",
        help="Not use environment variable to pass "
        "'local rank'. In atorch, the default value is False. "
        "If set to True, the script will pass "
        "--local_rank as argument.",
    )
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        "'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the training script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )
    parser.add_argument(
        "--coworker_size",
        type=int,
        default=-1,
        help="Launch `coworker_size` CPU pods to accelerate IO and " "preprocessing",
    )
    parser.add_argument(
        "--use_elastic_dataloader",
        default=False,
        action="store_true",
        help="Use coworker with elasticdl's dynamic data sharding",
    )
    parser.add_argument(
        "--rank_log_dir",
        type=str,
        default="",
        help="Log dir for each rank, format is path,mode. path is dir to dump log"
        "(if path is not exists, create it).mode is wb or ab, wb is overwrite, ab"
        "is append",
    )
    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training "
        "program/script to be launched in parallel, "
        "followed by all the arguments for the "
        "training script",
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)

    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args)


def wait_master_available(hostname, timeout=1800):
    elapse_time = 0

    while True:
        try:
            socket.gethostbyname(hostname)
            break
        except Exception:
            pass
        if elapse_time > timeout:
            return False
        time.sleep(10)
        elapse_time += 10

    return True


def cleanup(processes):
    for process, stdout, stderr in processes:
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()
        if process.poll() is None:
            process.terminate()


def check_process_loop(processes):
    running = True
    while running:
        running_process_ids = []
        failed_processes = []
        for idx, (process, stdout, stderr) in enumerate(processes):
            return_code = process.poll()
            if return_code is None:
                running_process_ids.append(idx)
            elif return_code != 0:
                if return_code < 0:
                    return_code = 128 - return_code
                failed_processes.append((idx, return_code))
        running = len(running_process_ids) > 0
        if len(failed_processes) > 0:
            for idx in running_process_ids:
                processes[idx][0].terminate()
            print("subprocess failed with LOCAL_RANK, return code: " + str(failed_processes))
            # use the first failed process return code as exit code
            exit_code = failed_processes[0][1]
            # if a signal kill
            exit(exit_code)
        if running:
            time.sleep(15)


def main(args):
    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    # coworker size in terms of number of cpu pods
    coworker_size = 0
    if "COWORKER_SIZE" in current_env:
        coworker_size = int(current_env["COWORKER_SIZE"])
    if args.coworker_size >= 0:
        coworker_size = args.coworker_size
    if coworker_size >= 0:
        current_env["COWORKER_SIZE"] = str(coworker_size)
    use_elastic_dataloader = args.use_elastic_dataloader
    if coworker_size == 0 and use_elastic_dataloader is True:
        use_elastic_dataloader = False
    current_env["USE_ELASTIC_DATALOADER"] = str(use_elastic_dataloader)

    # world size in terms of number of processes
    nnodes = 1
    if "WORLD_SIZE" in current_env:
        nnodes = int(current_env["WORLD_SIZE"])
    if args.nnodes > 0:
        nnodes = args.nnodes
    worker_size = nnodes - coworker_size
    dist_world_size = args.nproc_per_node * worker_size + coworker_size
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["NODE_SIZE"] = str(nnodes)
    current_env["NPROC_PER_NODE"] = str(args.nproc_per_node)

    # node rank
    node_rank = 0
    if "RANK" in current_env:
        node_rank = int(current_env["RANK"])
    if args.node_rank >= 0:
        node_rank = args.node_rank
    is_coworker_pod = node_rank >= worker_size
    is_gpu_pod = not is_coworker_pod

    if args.master_addr:
        current_env["MASTER_ADDR"] = args.master_addr
    if "MASTER_ADDR" not in current_env:
        current_env["MASTER_ADDR"] = "localhost"
    if args.master_port > 0:
        current_env["MASTER_PORT"] = str(args.master_port)
    if "MASTER_PORT" not in current_env:
        current_env["MASTER_PORT"] = "29500"
    if args.master_port2 > 0:
        current_env["MASTER_PORT2"] = str(args.master_port2)
    if "MASTER_PORT2" not in current_env:
        current_env["MASTER_PORT2"] = "29501"

    processes = []

    if "OMP_NUM_THREADS" not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print(
            "*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process "
            "to be {} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************".format(current_env["OMP_NUM_THREADS"])
        )

    # wait MASTER
    if node_rank != 0:
        master_ready = wait_master_available(current_env["MASTER_ADDR"])
        if not master_ready:
            print("master not ready after long wait, exiting!")
            exit(1)

    local_size = 1 if is_coworker_pod else args.nproc_per_node
    for local_rank in range(0, local_size):
        if is_coworker_pod:
            dist_rank = args.nproc_per_node * worker_size + node_rank - worker_size
            current_env["LOCAL_RANK"] = str(0)
        else:
            dist_rank = args.nproc_per_node * node_rank + local_rank
            current_env["LOCAL_RANK"] = str(local_rank)
        current_env["RANK"] = str(dist_rank)

        # spawn the processes
        with_python = not args.no_python
        cmd = []

        if args.nsys_profiling_rank == dist_rank:
            # launch the process with nsys
            cmd.append(args.nsys_command)
            cmd.extend(["launch", "-t", "cuda,cublas,cudnn", "-w", "true"])
            envs = "LOCAL_RANK={},RANK={},WORLD_SIZE={},".format(
                local_rank,
                dist_rank,
                dist_world_size,
            )
            envs += "MASTER_ADDR={},MASTER_PORT={}".format(current_env["MASTER_ADDR"], current_env["MASTER_PORT"])
            cmd.extend(["-e", envs])
            print("Starting rank={} with nsys".format(dist_rank), flush=True)

        if with_python:
            cmd.extend([sys.executable, "-u"])
            if args.module:
                cmd.append("-m")
        else:
            if args.not_use_env:
                raise ValueError("When using the '--no_python' flag, you must not set" " the '--not_use_env' flag.")
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag and the " "'--module' flag at the same time.")

        cmd.append(args.training_script)

        if args.not_use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(args.training_script_args)
        stdout = None
        stderr = None
        universal_newlines = False
        if args.rank_log_dir:
            split = args.rank_log_dir.split(",")
            mode = "wb"
            path = None
            if len(split) > 2:
                raise ValueError(f"wrong rank_log_dir, format is --rank_log_dir=/tmp/,a, your is {args.rank_log_dir}")
            elif len(split) == 1:
                path = split[0]
            else:
                path, mode = split
            from pathlib import Path

            log_dir = Path(path)
            if log_dir.exists() and not log_dir.is_dir():
                raise ValueError("rank_log_dir must be dir")
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout = open(f"{path}/{dist_rank}.stdout", mode)
            stderr = open(f"{path}/{dist_rank}.stderr", mode)

        if is_gpu_pod or use_elastic_dataloader is False:
            process = subprocess.Popen(
                cmd, env=current_env, stdout=stdout, stderr=stderr, universal_newlines=universal_newlines
            )
            processes.append((process, stdout, stderr))
        else:
            is_coworker0 = args.nproc_per_node * worker_size == dist_rank
            if is_coworker0:
                # coworker0 needs to initialize dynamic data sharding service
                free_port = find_free_port()
                coworker0_cmd = [
                    "python",
                    "-m",
                    "elasticdl.python.master.main",
                    "--job_name",
                    "dynamic_sharding_svc",
                    "--need_pod_manager",
                    "False",
                    "--distribution_strategy",
                    "Local",
                    "--task_fault_tolerance",
                    "False",
                    "--port",
                    "{}".format(free_port),
                ]
                process = subprocess.Popen(coworker0_cmd, env=current_env)
                store = dist.TCPStore(
                    current_env["MASTER_ADDR"],
                    int(current_env["MASTER_PORT2"]),
                    dist_world_size,
                    False,
                    timeout=timedelta(seconds=900),
                )
                ip_and_port = get_ip_address() + ":" + str(free_port)
                store.set(str(dist_rank), ip_and_port)
            else:
                process = subprocess.Popen(
                    cmd, env=current_env, stdout=stdout, stderr=stderr, universal_newlines=universal_newlines
                )
                processes.append((process, stdout, stderr))

    # terminate running subprocess to avoid zombie subprocess.
    atexit.register(cleanup, processes)

    check_process_loop(processes)


if __name__ == "__main__":
    args = parse_args()
    main(args)
