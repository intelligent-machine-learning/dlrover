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


import logging
import os
import time

import torch
import torch.distributed
from codetiming import Timer
from megatron.core import parallel_state as mpu
from omegaconf import DictConfig
from verl import DataProto
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.megatron_checkpoint_manager import (
    MegatronCheckpointManager,
)
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.utils.model import (
    load_mcore_dist_weights,
    load_megatron_gptmodel_weights,
)
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.megatron_workers import set_random_seed
from verl.workers.reward_model.megatron.reward_model import MegatronRewardModel

from dlrover.python.unified.trainer.example.verl.base.decorator import (
    Dispatch,
    collect_megatron_compute_data_proto,
    collect_megatron_pp_as_dp_data_proto,
    dispatch_megatron_compute_data_proto,
    dispatch_megatron_pp_as_dp_data_proto,
    dispatch_one_to_all,
    register,
)
from dlrover.python.unified.trainer.example.verl.base.worker import (
    MegatronWorker,
)
from dlrover.python.unified.trainer.workload import trainer_invocation

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ActorRolloutRefWorker(MegatronWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def init(self):
        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

            if self.config.actor.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.actor.megatron.context_parallel_size,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.actor.megatron.seed)

        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # normalize config
        # role is separated in dlrover
        if self.is_actor_or_rollout_device_collocation():
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= (
                mpu.get_data_parallel_world_size()
            )
            if self.config.actor.get("ppo_micro_batch_size", None):
                self.config.actor.ppo_micro_batch_size //= (
                    mpu.get_data_parallel_world_size()
                )
                self.config.rollout.log_prob_micro_batch_size //= (
                    mpu.get_data_parallel_world_size()
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = (
                    self.config.actor.ppo_micro_batch_size
                )
                self.config.rollout.log_prob_micro_batch_size_per_gpu = (
                    self.config.rollout.log_prob_micro_batch_size
                )

            self._is_offload_param = self.config.actor.megatron.get(
                "param_offload", False
            )
            self._is_offload_grad = self.config.actor.megatron.get(
                "grad_offload", False
            )
            self._is_offload_optimizer = self.config.actor.megatron.get(
                "optimizer_offload", False
            )
        elif self.is_ref_role():
            if self.config.ref.get("ppo_micro_batch_size", None):
                self.config.ref.log_prob_micro_batch_size //= (
                    mpu.get_data_parallel_world_size()
                )
                self.config.ref.ppo_micro_batch_size_per_gpu = (
                    self.config.ref.ppo_micro_batch_size
                )
            self._ref_is_offload_param = self.config.ref.megatron.get(
                "param_offload", False
            )

    def _build_model_optimizer(
        self, model_path, optim_config, override_model_config
    ):
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from verl.utils.megatron_utils import (
            get_model,
            init_megatron_optim_config,
        )
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(
            model_path, self.dtype, override_model_config
        )
        self.generation_config = get_generation_config(self.local_path)

        def megatron_actor_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(
                self.tf_config,
                self.hf_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                value=False,
            )
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        if self.is_actor_or_rollout_device_collocation():
            actor_module = get_model(
                megatron_actor_model_provider,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            print(f"actor_module: {len(actor_module)}")
            if self.config.actor.load_weight:
                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        actor_module,
                        self.config.actor.megatron.dist_checkpointing_path,
                        is_value_model=False,
                    )
                else:
                    load_megatron_gptmodel_weights(
                        self.config,
                        self.hf_config,
                        actor_module,
                        params_dtype=self.dtype,
                        is_value_model=False,
                    )

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
        elif self.is_ref_role():
            print(
                f"self.config.ref.load_weight: {self.config.ref.load_weight}"
            )
            ref_module = get_model(
                model_provider_func=megatron_actor_model_provider,
                model_type=ModelType.encoder_or_decoder,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )

            if self.config.ref.load_weight:  # should align with the actor:
                assert (
                    self.config.actor.load_weight
                    == self.config.ref.load_weight
                )
                print("load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        ref_module,
                        self.config.ref.megatron.dist_checkpointing_path,
                        is_value_model=False,
                    )
                else:
                    load_megatron_gptmodel_weights(
                        self.config,
                        self.hf_config,
                        ref_module,
                        params_dtype=self.dtype,
                        is_value_model=False,
                    )
            log_gpu_memory_usage("After ref module init", logger=logger)
            return ref_module, self.hf_config

        if self.is_actor_role():
            optim_config = init_megatron_optim_config(optim_config)
            actor_optimizer = get_megatron_optimizer(
                model=actor_module, config=optim_config
            )
        else:
            optim_config = None
            actor_optimizer = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return actor_module, actor_optimizer, self.hf_config, optim_config

    def _build_rollout(self, trust_remote_code=False):
        if self.config.rollout.name == "vllm":
            from torch.distributed.device_mesh import init_device_mesh
            from verl.workers.rollout.vllm_rollout import (
                vllm_mode,
                vLLMRollout,
            )
            from verl.workers.sharding_manager.megatron_vllm import (
                MegatronVLLMShardingManager,
            )

            # NOTE(sgm): If the QKV and gate_up projection layer are concate together in actor,
            # we will reorganize their weight format when resharding from actor to rollout.
            layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.weight",
            }

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert (
                self.world_size % infer_tp == 0
            ), f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            rollout_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, infer_tp),
                mesh_dim_names=["dp", "infer_tp"],
            )
            log_gpu_memory_usage("Before building vllm rollout", logger=None)

            local_path = copy_to_local(self.config.model.path)
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                )
            elif vllm_mode == "spmd":
                rollout = vLLMRollout(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                )
            log_gpu_memory_usage("After building vllm rollout", logger=logger)

            # perform weight resharding between actor and rollout
            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(
                self.actor_model_config, self.dtype
            )
            sharding_manager = MegatronVLLMShardingManager(
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                layer_name_mapping=layer_name_mapping,
                actor_module=self.actor.actor_module,
                weight_converter=weight_converter,
            )
            log_gpu_memory_usage(
                "After building sharding manager", logger=logger
            )
        else:
            raise NotImplementedError(
                "Only vllmRollout is supported with Megatron now"
            )

        return rollout, sharding_manager

    @trainer_invocation(pre_func=dispatch_one_to_all)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from omegaconf import OmegaConf
        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )
        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage(
            "Before init actor model and optimizer", logger=logger
        )
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        if self.is_actor_role() or self.is_rollout_role():
            # we need the model for actor and rollout
            optim_config = self.config.actor.optim if self._is_actor else None
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_model_config,
                self.actor_optim_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
            )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage(
                    "After offload actor params and grad during init",
                    logger=logger,
                )
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )

        if self.is_actor_role:
            self.actor = MegatronPPOActor(
                config=self.config.actor,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        if self.is_rollout_role:
            self.rollout, self.sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get(
                    "trust_remote_code", False
                )
            )
            log_gpu_memory_usage("After rollout init", logger=logger)

        if self.is_ref_role():
            (
                self.ref_module,
                self.ref_model_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage(
                    "After offload ref params during init", logger=logger
                )

        if self.is_actor_role():
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                model_config=self.actor_model_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                tokenizer=self.tokenizer,
                optimizer=self.actor_optimizer,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    @trainer_invocation(
        pre_func=dispatch_megatron_compute_data_proto,
        post_func=collect_megatron_compute_data_proto,
    )
    @GPUMemoryLogger(role="update_actor", logger=logger)
    def update_actor(self, data: DataProto):
        assert self.is_actor_role()
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage(
                "After load actor params and grad during update_actor",
                logger=logger,
            )
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage(
                "After load actor optimizer during update_actor", logger=logger
            )
        data.batch = data.batch.cuda()

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            global_num_tokens, delta_time
        )
        metrics["perf/mfu/actor"] = (
            estimated_flops
            * self.config.actor.ppo_epochs
            / promised_flops
            / self.world_size
        )

        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage(
                "After offload actor params and grad during update_actor",
                logger=logger,
            )
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage(
                "After offload actor optimizer during update_actor",
                logger=logger,
            )

        torch.cuda.empty_cache()
        return output

    @trainer_invocation(
        pre_func=dispatch_megatron_pp_as_dp_data_proto,
        post_func=collect_megatron_pp_as_dp_data_proto,
    )
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    def generate_sequences(self, prompts: DataProto):
        assert self.is_rollout_role()
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage(
                "After load actor params during generate_sequences",
                logger=logger,
            )
        prompts.batch = prompts.batch.cuda()
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.sharding_manager:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage(
                "After entering sharding manager", logger=logger
            )

            prompts = self.sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        # clear kv cache
        torch.cuda.empty_cache()
        return output

    @trainer_invocation(
        pre_func=dispatch_megatron_compute_data_proto,
        post_func=collect_megatron_compute_data_proto,
    )
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    def compute_ref_log_prob(self, data: DataProto):
        data = data.to("cuda")
        assert self.is_ref_role()
        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            log_gpu_memory_usage(
                "After load ref params and grad during compute_ref_log_prob",
                logger=logger,
            )
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        output, _ = self.ref_policy.compute_log_prob(
            data=data, calculate_entropy=False
        )
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
        output = output.to("cpu")
        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
            log_gpu_memory_usage(
                "After offload ref params and grad during compute_ref_log_prob",
                logger=logger,
            )
        torch.cuda.empty_cache()
        return output

    @trainer_invocation(
        pre_func=dispatch_megatron_compute_data_proto,
        post_func=collect_megatron_compute_data_proto,
    )
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    def compute_log_prob(self, data: DataProto):
        assert self.is_actor_role()
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage(
                "After load actor params and grad during compute_log_prob",
                logger=logger,
            )
        data = data.to("cuda")
        output = data
        # we should always recompute old_log_probs when it is HybridEngine
        output.meta_info[
            "micro_batch_size"
        ] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        output.meta_info["temperature"] = self.config.rollout.temperature
        old_log_probs, entropys = self.actor.compute_log_prob(
            data=output, calculate_entropy=True
        )
        output.batch["old_log_probs"] = old_log_probs
        output.batch["entropys"] = entropys
        output = output.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage(
                "After offload actor params and grad during compute_log_prob",
                logger=logger,
            )
        torch.cuda.empty_cache()
        return output

    @trainer_invocation(pre_func=dispatch_one_to_all)
    def load_checkpoint(
        self, checkpoint_path, hdfs_path=None, del_local_after_load=True
    ):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    @trainer_invocation(pre_func=dispatch_one_to_all)
    def load_pretrained_model(
        self, checkpoint_path, del_local_after_load=True
    ):
        pass

    @trainer_invocation(pre_func=dispatch_one_to_all)
    def save_checkpoint(
        self,
        checkpoint_path,
        hdfs_path=None,
        global_step=0,
        max_ckpt_to_keep=None,
    ):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)


class CriticWorker(MegatronWorker):
    def init(self):
        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.get("ppo_micro_batch_size", None):
            self.config.ppo_micro_batch_size //= (
                mpu.get_data_parallel_world_size()
            )
            self.config.ppo_micro_batch_size_per_gpu = (
                self.config.ppo_micro_batch_size
            )

    def _build_critic_model_optimizer(
        self, model_path, optim_config, override_model_config
    ):
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from verl.utils.megatron_utils import (
            get_model,
            init_megatron_optim_config,
        )
        from verl.utils.model import print_model_size

        self._init_hf_config_and_tf_config(
            model_path, self.dtype, override_model_config
        )

        def megatron_critic_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(
                self.tf_config,
                self.hf_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=False,
                value=True,
            )
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        critic_module = get_model(
            model_provider_func=megatron_critic_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    critic_module,
                    self.config.megatron.dist_checkpointing_path,
                    is_value_model=True,
                )
            else:
                load_megatron_gptmodel_weights(
                    self.config,
                    self.hf_config,
                    critic_module,
                    params_dtype=self.dtype,
                    is_value_model=True,
                )
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        optim_config = init_megatron_optim_config(optim_config)
        critic_optimizer = get_megatron_optimizer(
            model=critic_module, config=optim_config
        )
        torch.cuda.empty_cache()
        return critic_module, critic_optimizer, self.hf_config, optim_config

    @trainer_invocation(pre_func=dispatch_one_to_all)
    def init_model(self):
        # create critic
        from omegaconf import OmegaConf
        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        (
            self.critic_module,
            self.critic_optimizer,
            self.critic_model_config,
            critic_optimizer_config,
        ) = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            model_config=self.critic_model_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            tokenizer=self.tokenizer,
            optimizer=self.critic_optimizer,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )

    @trainer_invocation(
        pre_func=dispatch_megatron_compute_data_proto,
        post_func=collect_megatron_compute_data_proto,
    )
    def compute_values(self, data: DataProto):
        data = data.to("cuda")
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        return output

    @trainer_invocation(
        pre_func=dispatch_megatron_compute_data_proto,
        post_func=collect_megatron_compute_data_proto,
    )
    def update_critic(self, data: DataProto):
        data = data.to("cuda")

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(optimizer=self.critic_optimizer)

        dataloader = self.critic.make_minibatch_iterator(data)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(
            global_num_tokens, delta_time
        )
        metrics["perf/mfu/critic"] = (
            estimated_flops
            * self.config.ppo_epochs
            / promised_flops
            / self.world_size
        )
        output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)
        output = output.to("cpu")
        return output

    @trainer_invocation(pre_func=dispatch_one_to_all)
    @trainer_invocation()
    def load_checkpoint(
        self, checkpoint_path, hdfs_path=None, del_local_after_load=True
    ):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

    @trainer_invocation(pre_func=dispatch_one_to_all)
    def save_checkpoint(
        self,
        checkpoint_path,
        hdfs_path=None,
        global_steps=0,
        max_ckpt_to_keep=None,
    ):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path,
            hdfs_path=hdfs_path,
            global_step=global_steps,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)


class RewardModelWorker(MegatronWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def init(self):
        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_rm_model(self, model_path, override_model_config):
        from megatron.core.models.gpt.gpt_model import ModelType
        from verl.utils.megatron_utils import get_model

        self._init_hf_config_and_tf_config(
            model_path, self.dtype, override_model_config
        )

        def megatron_rm_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(
                self.tf_config,
                self.hf_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=False,
                value=True,
            )
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        reward_model = get_model(
            model_provider_func=megatron_rm_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # reward_model = nn.ModuleList(reward_model)

        if self.config.load_weight:
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    reward_model,
                    self.config.megatron.dist_checkpointing_path,
                    is_value_model=True,
                )
            else:
                load_megatron_gptmodel_weights(
                    self.config,
                    self.hf_config,
                    reward_model,
                    params_dtype=self.dtype,
                    is_value_model=True,
                )

        torch.cuda.empty_cache()
        return reward_model, self.hf_config

    @trainer_invocation(pre_func=dispatch_one_to_all)
    def init_model(self):
        # create critic
        from omegaconf import OmegaConf
        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        sft_tokenizer_local_path = copy_to_local(
            self.config.model.input_tokenizer
        )
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        rm_tokenizer_path = self.config.model.get("rm_tokenizer", None)
        rm_tokenizer = None
        if rm_tokenizer_path is not None:
            rm_tokenizer_local_path = copy_to_local(rm_tokenizer_path)
            rm_tokenizer = hf_tokenizer(rm_tokenizer_local_path)

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        reward_model_module, reward_model_config = self._build_rm_model(
            model_path=self.config.model.path,
            override_model_config=override_model_config,
        )
        # should be implemented in workers
        self.rm = MegatronRewardModel(
            config=self.config,
            reward_model_module=reward_model_module,
            model_config=reward_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            sft_tokenizer=sft_tokenizer,
            rm_tokenizer=rm_tokenizer,
        )

    # the input_ids, responses, attention_mask and position_ids may be different!

    @trainer_invocation(
        pre_func=dispatch_megatron_compute_data_proto,
        post_func=collect_megatron_compute_data_proto,
    )
    def compute_rm_score(self, data: DataProto):
        data.batch = data.batch.cuda()
        output = self.rm.compute_reward(data)
        output = output.to("cpu")
        return output
