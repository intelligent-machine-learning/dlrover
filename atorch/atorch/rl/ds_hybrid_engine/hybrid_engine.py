import os

import deepspeed
import torch
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.module_inject.layers import EmbeddingLayer, LinearLayer, Normalize, OPTEmbedding
from deepspeed.runtime.hybrid_engine import DeepSpeedHybridEngine
from torch import nn

from .ds_hook import *  # NOQA
from .module_inject.utils import policy_to_ds_container
from .replace_policy import replace_policies

try:
    import transformers

    OPTLearnedPositionalEmbedding = transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding
except ImportError:
    OPTLearnedPositionalEmbedding = None


import gc
import math
import time

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import get_inactive_params
from deepspeed.runtime.zero import GatheredParameters


def atorch_generate(self, *inputs, **kwargs):
    inputs = inputs[0]

    self._t0 = time.time()
    if self._total_batch_size is None:
        bsz = len(inputs)
        self._total_batch_size = bsz * dist.get_world_size()
    if self.Z3_enabled:
        with deepspeed.zero.GatheredParameters(list(self.module.parameters(recurse=True))):
            self._gather_latency = time.time() - self._t0
            vllm_backend = getattr(self, "vllm_backend", None)
            s1 = time.time()
            vllm_backend.set_train_model_weights(self.module)
            s2 = time.time()
            s3 = time.time()
            res = vllm_backend.generate(inputs)
            s4 = time.time()
            import os

            rank = int(os.environ.get("LOCAL_RANK", 0))
            if rank == 0:
                print("set weight cost", s2 - s1, "generate cost ", s4 - s3)
            self._generate_latency = time.time() - self._t0 - self._gather_latency
    else:
        self._gather_latency = 0
        vllm_backend = getattr(self, "vllm_backend", None)
        vllm_backend.set_train_model_weights(self.module)
        res = vllm_backend.generate(inputs)
        self._generate_latency = time.time() - self._t0 - self._gather_latency
    for param in vllm_backend.llm.llm_engine.workers[0].model.parameters():
        param.data = torch.empty(0, dtype=param.dtype, device=param.device)
    torch.cuda.empty_cache()
    return res


def generate(self, *inputs, **kwargs):
    import os

    rank = int(os.environ.get("LOCAL_RANK", 0))
    _gather_latency = 0

    if self._total_batch_size is None:
        bsz = inputs[0].shape[0] if len(inputs) > 0 else kwargs["input_ids"].shape[0]
        self._total_batch_size = bsz * dist.get_world_size()

    self._t0 = time.time()

    if self.Z3_enabled and self.gather_all_layers:
        if self._config.hybrid_engine.inference_tp_size > 1:
            non_tp_params = []
            for other_layer in self._other_layers:
                non_tp_params.extend(list(other_layer.parameters()))

            partition_size = self._config.hybrid_engine.tp_gather_partition_size

            layer_groups = math.ceil(len(self.layer_params) / partition_size)
            for lg in range(layer_groups):
                non_active_params = []
                non_active_lora_params = []
                for layer_id in range(
                    lg * partition_size,
                    min(len(self.layer_params), (lg + 1) * partition_size),
                    1,
                ):
                    non_tp_params.extend(self.layer_params[layer_id][:4])
                    non_active_params.extend(get_inactive_params(self.layer_params[layer_id]))
                    non_active_params.extend(get_inactive_params(self.layer_lora_params[layer_id]))
                with GatheredParameters(non_active_params):
                    for layer_id in range(
                        lg * partition_size,
                        min(len(self.layer_params), (lg + 1) * partition_size),
                        1,
                    ):
                        if len(self.all_lora_params) > 0:
                            self._fuse_lora_layer(layer_id)

                        self._inference_containers[layer_id].apply_tensor_parallelism(
                            self.mp_replace, reversed_dim=True
                        )

            # TODO(cmikeh2) Evaluate if this can be deferred when release_inference_cache
            # is enabled.
            gc.collect()
            get_accelerator().empty_cache()

            _gather_latency = time.time() - self._t0

            input_shape = inputs[0].shape if len(inputs) > 0 else kwargs["input_ids"].shape
            output = torch.zeros(
                (input_shape[0] * self._config.hybrid_engine.inference_tp_size,) + input_shape[1:],
                dtype=inputs[0].dtype if len(inputs) > 0 else kwargs["input_ids"].dtype,
                device=inputs[0].device if len(inputs) > 0 else kwargs["input_ids"].device,
            )
            input_cont = inputs[0].contiguous() if len(inputs) > 0 else kwargs["input_ids"].contiguous()
            dist.all_gather_into_tensor(output, input_cont, group=self.mp_group)

            if kwargs.get("attention_mask", None) is not None:
                attention_mask = kwargs.get("attention_mask")
                new_atention_mask = torch.zeros(
                    (input_shape[0] * self._config.hybrid_engine.inference_tp_size,) + input_shape[1:],
                    dtype=inputs[0].dtype if len(inputs) > 0 else kwargs["input_ids"].dtype,
                    device=inputs[0].device if len(inputs) > 0 else kwargs["input_ids"].device,
                )
                attention_mask_cont = attention_mask.contiguous()
                dist.all_gather_into_tensor(new_atention_mask, attention_mask_cont, group=self.mp_group)

                kwargs["attention_mask"] = new_atention_mask.to(rank)

            # broadcast position id
            # position id is batch_size, 2, seq
            if kwargs.get("position_ids", None) is not None:
                glm_position_ids = kwargs.get("position_ids")
                glm_new_position_ids = torch.zeros(
                    (input_shape[0] * self._config.hybrid_engine.inference_tp_size,) + glm_position_ids.shape[1:],
                    dtype=inputs[0].dtype if len(inputs) > 0 else kwargs["input_ids"].dtype,
                    device=inputs[0].device if len(inputs) > 0 else kwargs["input_ids"].device,
                )
                glm_attention_mask_cont = glm_position_ids.contiguous()
                dist.all_gather_into_tensor(glm_new_position_ids, glm_attention_mask_cont, group=self.mp_group)
                import os

                rank = int(os.environ.get("LOCAL_RANK", 0))
                kwargs["position_ids"] = glm_new_position_ids.to(rank)

            # broadcast attention mask
            # attention maks is batch_size, seq, seq
            if kwargs.get("generation_attention_mask", None) is not None:
                glm_attention_mask = kwargs.get("generation_attention_mask")
                glm_attention_mask = glm_attention_mask.squeeze(1)

                glm_new_atention_mask = torch.zeros(
                    (input_shape[0] * self._config.hybrid_engine.inference_tp_size,) + glm_attention_mask.shape[1:],
                    dtype=inputs[0].dtype if len(inputs) > 0 else kwargs["input_ids"].dtype,
                    device=inputs[0].device if len(inputs) > 0 else kwargs["input_ids"].device,
                )
                glm_attention_mask_cont = glm_attention_mask.contiguous()
                dist.all_gather_into_tensor(glm_new_atention_mask, glm_attention_mask_cont, group=self.mp_group)
                import os

                rank = int(os.environ.get("LOCAL_RANK", 0))
                if rank == 0:
                    print("broad casting  generation_attention_mask")
                kwargs["generation_attention_mask"] = glm_new_atention_mask.unsqueeze(1).to(rank)

            if len(inputs) > 0:
                inputs = (output, *inputs[1:])
            else:
                kwargs["input_ids"] = output

            self.retake_inference_cache()

            non_active_params = get_inactive_params(non_tp_params)
            with GatheredParameters(non_active_params):
                generate_ret_vals = self._generate(*inputs, **kwargs)

            for layer_id in range(len(self.layer_params)):
                self._inference_containers[layer_id].release_memory()

            rank = dist.get_rank(group=self.mp_group)
            generate_ret_vals = generate_ret_vals[input_shape[0] * rank : input_shape[0] * (rank + 1)]

        else:
            non_active_layers = get_inactive_params(self.all_layers_params)
            non_active_lora_params = get_inactive_params(self.all_lora_params)
            non_active_layers.extend(non_active_lora_params)
            with GatheredParameters(non_active_layers):
                _gather_latency = time.time() - self._t0

                if len(self.all_lora_params) > 0:
                    self.fuse_lora_weight()

                self.retake_inference_cache()
                generate_ret_vals = self._generate(*inputs, **kwargs)

                if len(self.all_lora_params) > 0:
                    self.unfuse_lora_weight()
    else:
        if len(self.all_lora_params) > 0 and (not self.Z3_enabled):
            self.fuse_lora_weight()

        self.retake_inference_cache()
        generate_ret_vals = self._generate(*inputs, **kwargs)

        if len(self.all_lora_params) > 0:
            if not self.Z3_enabled:
                self.unfuse_lora_weight()
            else:
                self.unfuse_lora_weight_non_pinned()
            self.is_lora_fused = False

    if self._config.hybrid_engine.release_inference_cache:
        inference_cuda_module.release_workspace()  # NOQA
        gc.collect()
        get_accelerator().empty_cache()

    self._generate_latency += time.time() - self._t0 - _gather_latency
    self._gather_latency += _gather_latency

    return generate_ret_vals


def eval(self):

    if self._t_start is not None:
        latency = time.time() - self._t_start
        self._total_latency = self._total_latency + latency
        self._iters = self._iters + 1
        if not dist.is_initialized() or dist.get_rank() == 0:
            _ = latency - (self._generate_latency + self._training_latency)

        self._t_start = time.time()
    self._training_latency = 0
    self._generate_latency = 0
    self._gather_latency = 0
    super(DeepSpeedHybridEngine, self).eval()
    if len(self._inference_containers) > 0 and self.Z3_enabled:
        for i, (orig_module, inference_container) in enumerate(zip(self._orig_modules, self._inference_containers)):
            if self.Z3_enabled and not self.gather_all_layers:
                orig_module.forward = self._zero3_forward(i)
            else:
                orig_module.forward = inference_container.module.forward

            inference_container.transform_for_inference()

        if not self.Z3_enabled or self.gather_all_layers:
            for orig_module, inference_layer in zip(self._orig_modules_others, self._other_layers):
                orig_module.forward = inference_layer.forward
    if self.Z3_enabled:
        gc.collect()
        get_accelerator().empty_cache()
    if self._t_start is None:
        self._t_start = time.time()


DeepSpeedHybridEngine.generate = generate
DeepSpeedHybridEngine.atorch_generate = atorch_generate
DeepSpeedHybridEngine.eval = eval


class NewDeepSpeedHybridEngine(DeepSpeedHybridEngine):
    def __init__(self, args, model, **kwargs):
        super().__init__(args, model, **kwargs)

    def sync_weight(self):
        from atorch.rl.model_utils.llama2_utils import get_llama2_params_offsets, move_weight_to_continuous_buffer

        if not hasattr(self, "vllm_comm") or self.vllm_comm is None:
            return

        if hasattr(self, "vllm_comm"):
            # mock get last two layer
            if not hasattr(self, "trainable_paramters"):
                trainable_parameters = []
                trainable_paramter_names = []
                for name, parameter in self.named_parameters():
                    if parameter.requires_grad is True:
                        trainable_parameters.append(parameter)
                        trainable_paramter_names.append(name)
                self.trainable_parameters = trainable_parameters
                self.trainable_paramter_names = trainable_paramter_names

            if self.Z3_enabled:
                with deepspeed.zero.GatheredParameters(self.trainable_parameters):
                    state_dict = {}
                    for name, parameter in zip(self.trainable_paramter_names, self.trainable_parameters):
                        state_dict[name] = parameter
                    offsets_info = get_llama2_params_offsets(config=self.module.config, tp_size=1)
                    param_tensor = move_weight_to_continuous_buffer(state_dict, self.module.config, offsets_info)
                    rank = int(os.environ.get("RANK", 0))
                    if rank == 0:
                        if hasattr(self, "vllm_comm") and self.vllm_comm is not None:
                            self.vllm_comm.send_data(param_tensor)
            else:
                for name, parameter in zip(self.trainable_paramter_names, self.trainable_parameters):
                    state_dict[name] = parameter
                    offsets_info = get_llama2_params_offsets(config=self.module.config, tp_size=1)
                    param_tensor = move_weight_to_continuous_buffer(state_dict, self.module.config, offsets_info)
                    rank = int(os.environ.get("RANK", 0))
                    if rank == 0:
                        if hasattr(self, "vllm_comm") and self.vllm_comm is not None:
                            self.vllm_comm.send_data(param_tensor)

        return

    def new_inference_container(self, orig_layer, policy_cls, layer_id):
        policy = policy_cls(orig_layer, inference=True)

        if self._config.fp16_enabled:
            inference_dtype = torch.float16
        elif self._config.bfloat16_enabled:
            inference_dtype = torch.bfloat16
        else:
            inference_dtype = torch.float32

        _container = policy_to_ds_container(
            policy=policy,
            config=DeepSpeedInferenceConfig(
                set_empty_params=True,
                dtype=inference_dtype,
                max_out_tokens=self._config.hybrid_engine.max_out_tokens,
                min_out_tokens=self._config.hybrid_engine.max_out_tokens,
                transposed_mode=True,
                return_tuple=False,
            ),
            model_config=self.module.config if hasattr(self.module, "config") else None,
            layer_id=layer_id,
            child=orig_layer,
        )

        if self.mpu is not None:
            if hasattr(self.mpu, "get_model_parallel_world_size"):
                _container.set_tensor_parallel_config(
                    self.mpu.get_model_parallel_world_size(),
                    self.mpu.get_model_parallel_group(),
                )
            else:
                _container.set_tensor_parallel_config(
                    self.mpu.get_tensor_model_parallel_world_size(),
                    self.mpu.get_tensor_model_parallel_group(),
                )
        else:
            _container.set_tensor_parallel_config(self._config.hybrid_engine.inference_tp_size, self.mp_group)
        _container.initialize_tensors(enable_training=True)
        _container.create_ds_model_config()
        _container.create_module()
        _container.module.attention.input_nw = _container.input_nw
        _container.module.attention.input_nb = _container.input_nb
        _container.set_params_wo_copy(Z3_enabled=self.Z3_enabled)
        return _container

    def populate_all_inference_policies(self):
        self.inference_policies = {}
        for plcy in replace_policies:
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    self.inference_policies.update({orig_layer_class: (self.new_inference_container, plcy)})
            elif plcy._orig_layer_class is not None:
                self.inference_policies.update({plcy._orig_layer_class: (self.new_inference_container, plcy)})
        self.inference_policies.update(
            {
                nn.Linear: (LinearLayer,),
                nn.Embedding: (EmbeddingLayer,),
                nn.LayerNorm: (Normalize,),
                OPTLearnedPositionalEmbedding: (OPTEmbedding,),
            }
        )
