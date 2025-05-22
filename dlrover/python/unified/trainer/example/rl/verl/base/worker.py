# Copyright 2025 The DLRover Authors. All rights reserved.
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
#
# This package includes code from [https://github.com/volcengine/verl]
# licensed under the Apache License 2.0. See [https://github.com/volcengine/
# verl] for details.

import os
import socket
from dataclasses import dataclass

from omegaconf import DictConfig
from ray.actor import ActorHandle

from dlrover.python.unified.trainer.rl_workload import BaseRLWorkload
from dlrover.python.unified.trainer.workload import (
    trainer_invocation,
)
from .decorator import Dispatch, Execute, register


@dataclass
class DistRankInfo:
    tp_rank: int
    dp_rank: int
    pp_rank: int
    cp_rank: int


@dataclass
class DistGlobalInfo:
    tp_size: int
    dp_size: int
    pp_size: int
    cp_size: int


class WorkerHelper:
    def _get_node_ip(self):
        def get_node_ip_by_sdk():
            if os.getenv("WG_BACKEND", None) == "ray":
                import ray

                return ray._private.services.get_node_ip_address()
            else:
                raise NotImplementedError(
                    "WG_BACKEND now just support ray mode."
                )

        host_ipv4 = os.getenv("MY_HOST_IP", None)
        host_ipv6 = os.getenv("MY_HOST_IPV6", None)
        host_ip_by_env = host_ipv4 or host_ipv6
        host_ip_by_sdk = get_node_ip_by_sdk()

        host_ip = host_ip_by_env or host_ip_by_sdk
        return host_ip

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_availale_master_addr_port(self):
        return self._get_node_ip(), str(self._get_free_port())

    def _get_pid(self):
        return


class WorkerMeta:
    keys = [
        "WORLD_SIZE",
        "RANK",
        "LOCAL_WORLD_SIZE",
        "LOCAL_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "CUDA_VISIBLE_DEVICES",
    ]

    def __init__(self, store) -> None:
        self._store = store

    def to_dict(self):
        return {
            f"_{key.lower()}": self._store.get(f"_{key.lower()}", None)
            for key in WorkerMeta.keys
        }


# we assume that in each WorkerGroup, there is a Master Worker
class Worker(WorkerHelper, BaseRLWorkload):
    """A (distributed) worker."""

    fused_worker_attr_name = "fused_worker_dict"

    def __init__(self, master_handle: ActorHandle, config: DictConfig):
        super().__init__(master_handle, config)

        self.cuda_visible_devices = None

    def post_setup(self):
        store = {
            "_world_size": self.world_size,
            "_rank": self.rank,
            "_local_world_size": self.local_world_size,
            "_local_rank": self.local_rank,
            "_master_addr": self.torch_master_addr,
            "_master_port": self.torch_master_port,
        }

        meta = WorkerMeta(store=store)
        self._configure_with_meta(meta=meta)

    def _configure_with_meta(self, meta: WorkerMeta):
        """
        This function should only be called inside by WorkerGroup
        """
        assert isinstance(meta, WorkerMeta)
        self.__dict__.update(meta.to_dict())  # this is hacky
        for key in WorkerMeta.keys:
            val = self.__dict__.get(f"_{key.lower()}", None)
            if val is not None:
                os.environ[key] = str(val)
        os.environ["REDIS_STORE_SERVER_HOST"] = self.torch_master_addr

    def get_cuda_visible_devices(self):
        import os

        cuda_visible_devices = os.environ.get(
            "CUDA_VISIBLE_DEVICES", "not set"
        )
        return cuda_visible_devices

    @trainer_invocation(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    @register(
        dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO
    )
    def execute_func_rank_zero(self, func, *args, **kwargs):
        result = func(*args, **kwargs)
        return result


class MegatronWorker(Worker):
    @trainer_invocation(target="RANK0")
    def tp_size(self):
        return self.get_megatron_global_info().tp_size

    @trainer_invocation(target="RANK0")
    def dp_size(self):
        return self.get_megatron_global_info().dp_size

    @trainer_invocation(target="RANK0")
    def pp_size(self):
        return self.get_megatron_global_info().pp_size

    @trainer_invocation(target="RANK0")
    def cp_size(self):
        return self.get_megatron_global_info().cp_size

    @trainer_invocation(target="RANK0")
    def get_megatron_global_info(self):
        from megatron.core import parallel_state as mpu

        tp_size = mpu.get_tensor_model_parallel_world_size()
        dp_size = mpu.get_data_parallel_world_size()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        info = DistGlobalInfo(
            tp_size=tp_size, dp_size=dp_size, pp_size=pp_size, cp_size=cp_size
        )
        return info

    @trainer_invocation()
    def get_megatron_rank_info(self):
        from megatron.core import parallel_state as mpu

        tp_rank = mpu.get_tensor_model_parallel_rank()
        dp_rank = mpu.get_data_parallel_rank()
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        cp_rank = mpu.get_context_parallel_rank()
        info = DistRankInfo(
            tp_rank=tp_rank, dp_rank=dp_rank, pp_rank=pp_rank, cp_rank=cp_rank
        )
        return info

    def _init_hf_config_and_tf_config(
        self, model_path, dtype, override_model_config
    ):
        from transformers import AutoConfig
        from verl.models.mcore import hf_to_mcore_config
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(self.local_path)

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        self.share_embeddings_and_output_weights = getattr(
            hf_config, "tie_word_embeddings", False
        )
        update_model_config(
            hf_config, override_config_kwargs=override_config_kwargs
        )
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")
        tf_config = hf_to_mcore_config(hf_config, dtype)

        def add_optimization_config_to_tf_config(tf_config, verl_model_config):
            # add optimization config to tf_config, e.g. checkpointing
            if verl_model_config.get("enable_gradient_checkpointing", False):
                gradient_checkpointing_cfg = dict(
                    verl_model_config.get(
                        "gradient_checkpointing_kwargs", dict()
                    )
                )
                tf_config.recompute_method = gradient_checkpointing_cfg.get(
                    "activations_checkpoint_method", "full"
                )
                tf_config.recompute_granularity = (
                    gradient_checkpointing_cfg.get(
                        "activations_checkpoint_granularity", "full"
                    )
                )
                tf_config.recompute_num_layers = (
                    gradient_checkpointing_cfg.get(
                        "activations_checkpoint_num_layers", -1
                    )
                )

        add_optimization_config_to_tf_config(tf_config, self.config.model)

        print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config
