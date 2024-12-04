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

import psutil

from dlrover.python.common.constants import NodeEnv
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
