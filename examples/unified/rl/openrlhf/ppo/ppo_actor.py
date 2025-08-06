import math
import os
from functools import partial
from typing import List, Optional, Sequence

import deepspeed
import torch
import torch.distributed
from git import TYPE_CHECKING
from omegaconf import DictConfig
from openrlhf.models import Actor
from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.trainer.ray.ppo_actor import ActorPPOTrainer
from openrlhf.utils import get_tokenizer
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.deepspeed.deepspeed_utils import (
    offload_deepspeed_states,
    reload_deepspeed_states,
)
from openrlhf.utils.distributed_util import (
    torch_dist_barrier_and_cuda_sync,
)
from openrlhf.utils.logging_utils import init_logger
from transformers.optimization import get_scheduler

from dlrover.python.unified.api.runtime.rpc import rpc
from examples.unified.rl.openrlhf.ppo import remote_call

logger = init_logger(__name__)


class PolicyModelActor:
    @rpc(remote_call.actor_init)
    def init(
        self, strategy: DeepspeedStrategy, model_path: str, /, max_steps: int
    ):
        self.strategy = strategy
        strategy.setup_distributed()

        assert isinstance(strategy.args, DictConfig)
        args = self.args = strategy.args
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.max_steps = max_steps

        if getattr(args, "vllm_num_engines", 0) > 0:
            # To prevent hanging during NCCL synchronization of weights between DeepSpeed and vLLM.
            # see https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
            if getattr(args, "vllm_sync_backend", "nccl") == "nccl":
                os.environ["NCCL_CUMEM_ENABLE"] = "0"

        actor = Actor(
            model_path,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            packing_samples=strategy.args.packing_samples,
            temperature=strategy.args.temperature,
            use_liger_kernel=strategy.args.use_liger_kernel,
        )
        strategy.print(actor)

        # configure tokenizer
        self.tokenizer = get_tokenizer(
            model_path,
            actor.model,
            "left",
            strategy,
            use_fast=not strategy.args.disable_fast_tokenizer,
        )

        if args.enable_ema:
            ema_model = Actor(
                model_path,
                use_flash_attention_2=strategy.args.flash_attn,
                bf16=strategy.args.bf16,
                load_in_4bit=strategy.args.load_in_4bit,
                ds_config=strategy.get_ds_eval_config(offload=True),
                packing_samples=strategy.args.packing_samples,
            )
            ema_model._offload = True
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor,
            lr=args.actor_learning_rate,
            betas=strategy.args.adam_betas,
            weight_decay=args.l2,
        )

        actor_scheduler = get_scheduler(
            args.lr_scheduler,
            actor_optim,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={
                "min_lr": args.actor_learning_rate * 0.1
            },
        )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": args.gradient_checkpointing_use_reentrant
                }
            )

        # prepare models/optimizers...
        if TYPE_CHECKING:
            self.actor = actor
            self.actor_optim = actor_optim
            self.actor_scheduler = actor_scheduler
            self.ema_model = ema_model
        else:
            self.actor, self.actor_optim, self.actor_scheduler = (
                strategy.prepare(
                    (actor, actor_optim, actor_scheduler),
                    is_rlhf=True,
                )
            )
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)

        # load checkpoint
        self.checkpoint_states = {}
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            strategy.print(f"Loading the checkpoint: {ckpt_path}")
            _, states = strategy.load_ckpt(self.actor.model, ckpt_path)
            self.checkpoint_states["global_step"] = states["global_step"]
            self.checkpoint_states["episode"] = states["episode"]
            self.checkpoint_states["data_loader_state_dict"] = states[
                "data_loader_state_dict"
            ]

        # initial offload
        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)

        # configure Trainer
        self.trainer = ActorPPOTrainer(
            strategy,
            self.actor,
            ema_model=self.ema_model,
            actor_optim=self.actor_optim,
            actor_scheduler=self.actor_scheduler,
            micro_train_batch_size=args.micro_train_batch_size,
            tokenizer=self.tokenizer,
            eps_clip=args.eps_clip,
            ema_beta=args.ema_beta,
            vllm_engines=None,
        )

        self.init_vllm_sync_group()

    @rpc(remote_call.actor_train)
    def fit(self, kl_ctl: float = 0):
        """Train actor model with the replay buffer."""
        if self.args.deepspeed_enable_sleep:
            reload_deepspeed_states(self.actor.model)
        torch.cuda.empty_cache()
        self.actor.train()
        status = self.trainer.ppo_train(kl_ctl)
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if self.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.actor.model)
        return status

    def forward(
        self,
        sequences: torch.Tensor,
        action_mask: torch.BoolTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """Generates actor values."""
        device = torch.cuda.current_device()
        self.actor.eval()
        with torch.no_grad():
            action_log_probs = self.actor(
                sequences.to(device),
                action_mask.to(device),
                attention_mask.to(device),
                ring_attn_group=self.strategy.ring_attn_group,
            )
        self.actor.train()  # reset model state
        return action_log_probs.to("cpu")

    @rpc(remote_call.actor_forward)
    def batch_forward(
        self,
        sequences: Sequence[torch.Tensor],
        action_mask: Sequence[torch.BoolTensor],
        attention_mask: Sequence[torch.LongTensor],
    ) -> List[torch.Tensor]:
        return [
            self.forward(*args)
            for args in zip(sequences, action_mask, attention_mask)
        ]

    def init_vllm_sync_group(self):
        self.sync_use_ray = self.args.get("vllm_sync_with_ray", False)

        vllm_num_engines, vllm_tensor_parallel_size = (
            self.args.vllm_num_engines,
            self.args.vllm_tensor_parallel_size,
        )
        world_size = vllm_num_engines * vllm_tensor_parallel_size + 1
        backend = self.args.vllm_sync_backend
        group_name = "vllm_sync_group"
        import ray.util.collective as collective

        ref = remote_call.vllm_sync_setup_process_group(
            world_size, backend, group_name
        )
        collective.init_collective_group(
            world_size=world_size,
            rank=0,
            backend=backend,
            group_name=group_name,
        )
        ref.result()  # wait for vllm engines to finish
        broadcast = partial(
            collective.broadcast, src_rank=0, group_name=group_name
        )
        self._broadcast_to_vllm_one = broadcast

    @rpc(remote_call.actor_sync_to_vllm)
    def broadcast_to_vllm(self):
        """Broadcast actor model to vLLM engines."""
        torch.cuda.empty_cache()
        model: torch.nn.Module = self.actor.model.module  # type: ignore[assignment]
        count, num_params = 0, len(list(model.named_parameters()))

        is_rank0 = torch.distributed.get_rank() == 0

        def _broadcast_param(param, count, num_params):
            if not is_rank0:
                return
            # Fire all vllm engines for broadcast
            shape = (
                param.shape if self.args.zero_stage != 3 else param.ds_shape
            )
            ref = remote_call.vllm_sync_weight(name, param.dtype, shape)
            self._broadcast_to_vllm_one(param.data)
            ref.result()  # wait for vllm engines to finish

        if is_rank0:
            remote_call.vllm_sync_weight_begin()

        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            if self.args.ds_tensor_parallel_size > 1:
                with deepspeed.module_inject.layers.GatherReplacedLayerParams(
                    [param], model, enabled=True
                ):
                    _broadcast_param(param, count, num_params)
            else:
                with deepspeed.zero.GatheredParameters(
                    [param], enabled=self.args.zero_stage == 3
                ):
                    _broadcast_param(param, count, num_params)

        if is_rank0:
            remote_call.vllm_sync_weight_end()

        torch.cuda.empty_cache()
        torch_dist_barrier_and_cuda_sync()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    @rpc(remote_call.actor_append_experience)
    def append(self, experience: Experience):
        self.trainer.replay_buffer.append(experience)

    @rpc(remote_call.actor_save_model)
    def save_checkpoint(
        self, save_path: str, tag: Optional[str], ext_states: Optional[dict]
    ):
        args = self.args
        self.strategy.save_ckpt(
            self.actor.model,
            save_path,
            tag,
            args.max_ckpt_num,
            args.max_ckpt_mem,
            ext_states,
        )
        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(
                self.ema_model if args.enable_ema else self.actor,
                self.tokenizer,
                save_path,
            )
        # wait
        torch_dist_barrier_and_cuda_sync()
