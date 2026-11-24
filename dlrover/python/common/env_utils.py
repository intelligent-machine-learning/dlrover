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

import os
import socket
import subprocess
from typing import Tuple

import psutil

from dlrover.python.common.constants import CommunicationType, NodeEnv
from dlrover.python.common.log import default_logger as logger


def get_node_rank():
    """Get the node rank."""
    if NodeEnv.NODE_RANK in os.environ:
        rank = os.getenv(NodeEnv.NODE_RANK)
    else:
        # Backwards compatible env of dlrover elastic job with version < 0.3.0
        rank = os.getenv(NodeEnv.WORKER_RANK, "0")
    return int(rank)


def get_local_world_size():
    return int(os.getenv("LOCAL_WORLD_SIZE", 1))


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", 0))


def get_rank():
    return int(os.getenv("RANK", 0))


def get_group_world_size():
    return int(os.getenv("GROUP_WORLD_SIZE", 1))


def get_group_rank():
    return int(os.getenv("GROUP_RANK", 1))


def get_torch_restart_count():
    return int(os.getenv("TORCHELASTIC_RESTART_COUNT", 0))


def get_job_uid():
    """Get the job UID."""
    job_uid = os.getenv(NodeEnv.JOB_UID, "")
    return job_uid


def get_node_id():
    """Get the node ID."""
    node_id = int(os.getenv(NodeEnv.NODE_ID, 0))
    return node_id


def get_node_type():
    """Get the node type."""
    node_type = os.getenv(NodeEnv.NODE_TYPE, "worker")
    return node_type


def get_node_num():
    """Get the number of node."""
    node_num = int(os.getenv(NodeEnv.NODE_NUM, 0))
    return node_num


def get_env(env_key):
    """Get the specified environment variable."""
    env_value = os.getenv(env_key, None)
    return env_value


def set_env(env_key, env_value):
    """Set the specified environment variable."""
    os.environ[env_key] = str(env_value)


def print_process_list():
    """Print out the processes list"""
    try:
        for p in psutil.process_iter():
            name = " ".join(p.cmdline())
            logger.info(f"[{str(p.pid)}/{str(p.ppid())}] {name}")
    except Exception as e:
        logger.error(f"error in print process: {str(e)}")


def get_proc_env(pid):
    try:
        with open(f"/proc/{pid}/environ", "rb+") as f:
            data = f.read()
            envs = [chunk.decode("utf-8") for chunk in data.split(b"\x00")]
            return {
                k: v
                for k, v in (env.split("=", 1) for env in envs if "=" in env)
            }
    except Exception as e:
        logger.warning(f"Got error when getting process {pid} env: {str(e)}")
        return None


def is_worker_process(pid):
    envs = get_proc_env(pid)
    _has_run_id = False
    _has_master_addr = False
    _has_master_port = False

    if envs is not None:
        for k in envs.keys():
            if k == NodeEnv.TORCHELASTIC_RUN_ID:
                _has_run_id = True
            if k == NodeEnv.MASTER_ADDR:
                _has_master_addr = True
            if k == NodeEnv.MASTER_PORT:
                _has_master_port = True

    if _has_run_id and _has_master_addr and _has_master_port:
        return True
    else:
        return False


def get_all_child_pids(parent_pid=None):
    """Get all child pids by parent pid."""

    if parent_pid is None:
        parent_pid = os.getpid()

    child_pids = []

    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=False)
        child_pids = [child.pid for child in children]
    except psutil.NoSuchProcess:
        logger.warning(f"No such process {parent_pid}")
        return child_pids
    except Exception as e:
        logger.warning(f"Failed to get child processes by {parent_pid}, {e}")
        return child_pids

    return child_pids


def get_hostname_and_ip():
    """Get the hostname and IP address."""

    hostname = socket.gethostname()
    try:
        ip_address = socket.gethostbyname(hostname)
    except socket.error:
        ip_address = "Unknown"

    return hostname, ip_address


def is_ray_mode():
    if (
        get_env(NodeEnv.DLROVER_MASTER_SERVICE_TYPE)
        == CommunicationType.COMM_SERVICE_RAY
    ):
        return True
    return False


def get_kernel_stack(pid: int) -> Tuple[bool, str]:
    """
    Get kernel stack info (/proc/<pid>/stack)

    Args:
        pid: process id

    Returns:
        Tuple[result, stack info]
    """
    try:
        with open(f"/proc/{pid}/stack", "r") as f:
            stack_content = f.read()
        return True, stack_content
    except FileNotFoundError:
        error_msg = f"not exist: /proc/{pid}/stack"
        logger.warning(error_msg)
        return False, error_msg
    except PermissionError:
        error_msg = f"permission denied for: /proc/{pid}/stack"
        logger.warning(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"unexpected error when reading: /proc/{pid}/stack, {e}"
        logger.warning(error_msg)
        return False, error_msg


def get_user_stack_pyspy(pid: int) -> Tuple[bool, str]:
    """
    Use py-spy get stack info.

    Args:
        pid: process id

    Returns:
        Tuple[result, stack info]
    """
    cmd = ["py-spy", "dump", "--native", "--pid", str(pid)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "py-spy timed out"
    except FileNotFoundError:
        return False, "py-spy not installed"
    except Exception as e:
        return False, f"unexpected error: {e}"


def dump_cuda_kernel_trace(pid: int, timeout: int = 120) -> Tuple[bool, str]:
    """
    Use cuda-gdb to dump the CUDA kernel info (devices, contexts, kernels,
    threads and host stacks) of a training process. The dump is attached in
    batch mode and detaches afterwards, so the target process resumes
    running after the dump. On timeout, the cuda-gdb process is force-
    killed with SIGKILL (kill -9); the partial output already produced is
    kept, since the most valuable backtraces are printed before the hung
    `info cuda` command stalls.

    Args:
        pid: process id of the training process
        timeout: max seconds the cuda-gdb process can run (default 120s)

    Returns:
        Tuple[result, cuda-gdb output or error info]
    """
    cmd = [
        "cuda-gdb",
        "--quiet",
        "--nx",
        "--batch",
        "-ex",
        "set pagination off",
        "-ex",
        "set confirm off",
        "-ex",
        f"attach {pid}",
        "-ex",
        "info cuda devices",
        "-ex",
        "info cuda contexts",
        "-ex",
        "info cuda kernels",
        "-ex",
        "info cuda threads",
        "-ex",
        "thread apply all bt 12",
        "-ex",
        "detach",
        "-ex",
        "quit",
    ]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return False, "cuda-gdb not installed"
    except Exception as e:
        return False, f"unexpected error: {e}"

    try:
        output, stderr = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        # Force-kill (SIGKILL) the hung cuda-gdb and keep the partial
        # output produced before the timeout.
        process.kill()
        output, stderr = process.communicate()
        timed_out = True

    if timed_out:
        error = f"cuda-gdb timed out (pid={pid}, timeout={timeout})"
        if stderr:
            error = f"{error}\n{stderr}"
        if output:
            return False, f"{output}\n{error}"
        return False, error
    if process.returncode == 0:
        return True, output
    if output and stderr:
        output = f"{output}\n{stderr}"
    else:
        output = output or stderr
    return False, output if output else f"cuda-gdb rc={process.returncode}"
