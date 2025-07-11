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
# This package includes code from [https://github.com/OpenRLHF/OpenRLHF]
# licensed under the Apache License 2.0. See [https://github.com/OpenRLHF/
# OpenRLHF] for details.
import argparse
import time
from datetime import datetime

from openrlhf.cli.train_ppo_ray import _validate_args

from dlrover.python.common.log import default_logger as logger
from dlrover.python.unified.api.rl import RLJobBuilder
from dlrover.python.unified.util.config_util import args_2_omega_conf


def submit(args):
    _validate_args(args)

    actor_per_node = args.actor_num_gpus_per_node
    actor_total = args.actor_num_nodes * actor_per_node

    has_rollout = False
    rollout_total = 0
    rollout_per_node = 0
    if args.vllm_num_engines is not None and args.vllm_num_engines > 0:
        if args.colocate_all_models:
            assert (
                args.actor_num_nodes * args.actor_num_gpus_per_node
                == args.vllm_num_engines * args.vllm_tensor_parallel_size
            ), (
                f"actor_num_nodes * actor_num_gpus_per_node must be equal to "
                "vllm_num_engines * vllm_tensor_parallel_size, "
                f"got {args.actor_num_nodes * args.actor_num_gpus_per_node} "
                f"and {args.vllm_num_engines * args.vllm_tensor_parallel_size}"
            )
        has_rollout = True
        rollout_total = args.vllm_num_engines
        rollout_per_node = args.vllm_num_engines // args.node_num

    has_ref = False
    ref_total = 0
    ref_per_node = 0
    if args.init_kl_coef != 0:
        has_ref = True
        ref_per_node = args.ref_num_gpus_per_node
        ref_total = args.ref_num_nodes * ref_per_node

    has_critic = False
    critic_total = 0
    critic_per_node = 0
    if args.critic_pretrain:
        has_critic = True
        critic_per_node = args.critic_num_gpus_per_node
        critic_total = args.critic_num_nodes * critic_per_node

    has_reward = False
    reward_total = 0
    reward_per_node = 0
    if not args.remote_rm_url:
        has_reward = True
        # support only 1 reward pretrain for now
        reward_per_node = args.reward_num_gpus_per_node
        reward_total = args.reward_num_nodes * reward_per_node

    rl_job_builder = (
        RLJobBuilder()
        .node_num(args.node_num)
        .device_per_node(args.device_per_node)
        .config(args_2_omega_conf(args))
        .trainer(
            "dlrover.python.rl.trainer.default.openrlhf.ppo.ppo_trainer",
            "PPOTrainer",
        )
        .actor(
            "dlrover.python.rl.trainer.default.openrlhf.ppo.ppo_actor",
            "ActorModelRayActor",
        )
        .total(actor_total)
        .per_node(actor_per_node)
    )

    if has_rollout:
        (
            rl_job_builder.rollout(
                "dlrover.python.rl.trainer.default.openrlhf.ppo.ppo_rollout",
                "RolloutRayActor",
            )
            .total(rollout_total)
            .per_node(rollout_per_node)
            .enable_ray_auto_visible_device()
        )

    if has_ref:
        (
            rl_job_builder.reference(
                "dlrover.python.rl.trainer.default.openrlhf.ppo.ppo_base",
                "ReferenceModelRayActor",
            )
            .total(ref_total)
            .per_node(ref_per_node)
        )

    if has_critic:
        (
            rl_job_builder.critic(
                "dlrover.python.rl.trainer.default.openrlhf.ppo.ppo_critic",
                "CriticModelRayActor",
            )
            .total(critic_total)
            .per_node(critic_per_node)
        )

    if has_reward:
        (
            rl_job_builder.reward(
                "dlrover.python.rl.trainer.default.openrlhf.ppo.ppo_base",
                "RewardModelRayActor",
            )
            .total(reward_total)
            .per_node(reward_per_node)
        )

    if args.colocate_all_models:
        rl_job_builder.with_collocation_all()
    else:
        if args.colocate_actor_ref:
            rl_job_builder.with_collocation("actor", "reference")
        if args.colocate_critic_reward:
            rl_job_builder.with_collocation("critic", "reward")

    rl_job = rl_job_builder.build()
    rl_job.submit(job_name=args.job_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_job_name = str(int(time.time()))
    parser.add_argument(
        "--job_name",
        type=str,
        default=default_job_name,
        help="name of the job",
    )
    parser.add_argument("--node_num", type=int, default=1, help="node number")
    parser.add_argument(
        "--device_per_node", type=int, default=8, help="device number per node"
    )

    # Ray and vLLM
    parser.add_argument(
        "--ref_num_nodes",
        type=int,
        default=1,
        help="number of nodes for reference",
    )
    parser.add_argument(
        "--ref_num_gpus_per_node",
        type=int,
        default=8,
        help="number of gpus per node for reference",
    )
    parser.add_argument(
        "--reward_num_nodes",
        type=int,
        default=1,
        help="number of nodes for reward model",
    )
    parser.add_argument(
        "--reward_num_gpus_per_node",
        type=int,
        default=8,
        help="number of gpus per node for reward model",
    )
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="whether to colocate reference and actor model, if true, "
        "they will share same gpus.",
    )

    parser.add_argument(
        "--actor_num_nodes",
        type=int,
        default=1,
        help="number of nodes for actor",
    )
    parser.add_argument(
        "--actor_num_gpus_per_node",
        type=int,
        default=8,
        help="number of gpus per node for actor",
    )
    parser.add_argument(
        "--critic_num_nodes",
        type=int,
        default=1,
        help="number of nodes for critic",
    )
    parser.add_argument(
        "--critic_num_gpus_per_node",
        type=int,
        default=8,
        help="number of gpus per node for critic",
    )
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="whether to colocate critic and reward model, if true, "
        "they will share same gpus.",
    )
    parser.add_argument(
        "--colocate_all_models",
        action="store_true",
        default=False,
        help="whether to colocate all models (including vLLM engines), "
        "if true, they will share same gpus.",
    )

    # optional vLLM for text generation
    parser.add_argument(
        "--vllm_num_engines",
        type=int,
        default=None,
        help="number of vLLM Engines, set to 0 to disable vLLM",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument(
        "--vllm_sync_backend",
        type=str,
        default="nccl",
        help="DeepSpeed -> vLLM weight sync backend",
    )
    parser.add_argument(
        "--vllm_sync_with_ray", action="store_true", default=False
    )
    parser.add_argument(
        "--enable_prefix_caching", action="store_true", default=False
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        default=False,
        help="Disable CUDA graph in vLLM",
    )
    parser.add_argument(
        "--vllm_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for vLLM when using --colocate_all_models",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization",
    )

    # Checkpoints
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument(
        "--ckpt_path", type=str, default="./ckpt/checkpoints_ppo_ray"
    )
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument(
        "--disable_ds_ckpt", action="store_true", default=False
    )
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument(
        "--load_checkpoint", action="store_true", default=False
    )

    # DeepSpeed
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for deepspeed"
    )
    parser.add_argument(
        "--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage"
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=False
    )
    parser.add_argument("--torch_compile", action="store_true", default=False)
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="Enable bfloat16"
    )
    # Make EMA as an optional feature
    parser.add_argument(
        "--enable_ema",
        action="store_true",
        help="Enable EMA checkpoint for the model.",
    )
    parser.add_argument(
        "--zpg", type=int, default=1, help="ZeRO++ max partition size"
    )
    parser.add_argument(
        "--adam_offload",
        action="store_true",
        default=False,
        help="Offload Adam Optimizer",
    )
    parser.add_argument(
        "--actor_init_on_gpu", action="store_true", default=False
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        default=False,
        help="Enable FlashAttention2",
    )
    parser.add_argument(
        "--use_liger_kernel",
        action="store_true",
        default=False,
        help="Enable Liger Kernel",
    )
    parser.add_argument(
        "--grad_accum_dtype",
        type=str,
        default=None,
        help="Adam grad accum data type",
    )
    parser.add_argument("--overlap_comm", action="store_true", default=False)
    parser.add_argument(
        "--gradient_checkpointing_use_reentrant",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--disable_fast_tokenizer", action="store_true", default=False
    )
    parser.add_argument(
        "--deepspeed_enable_sleep",
        action="store_true",
        default=False,
        help="Enable sleep mode for deepspeed when using "
        "--colocate_all_models",
    )

    # packing samples using Flash Attention2
    parser.add_argument(
        "--packing_samples", action="store_true", default=False
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--target_modules", type=str, nargs="*", default="all-linear"
    )
    parser.add_argument("--lora_dropout", type=float, default=0)

    # PPO
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=1024)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument(
        "--prompt_max_len",
        type=int,
        default=1024,
        help="Max tokens for each prompt",
    )
    parser.add_argument(
        "--generate_max_len",
        type=int,
        default=1024,
        help="Max tokens to generate in PPO",
    )
    parser.add_argument(
        "--max_len", type=int, default=None, help="deprecated max_len"
    )
    parser.add_argument(
        "--max_samples", type=int, default=1e8, help="Max number of samples"
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Gradient clipping"
    )
    parser.add_argument(
        "--l2", type=float, default=0.0, help="weight decay loss"
    )
    parser.add_argument(
        "--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef"
    )
    parser.add_argument(
        "--eps_clip", type=float, default=0.2, help="PPO clip range"
    )
    parser.add_argument(
        "--value_clip", type=float, default=0.2, help="PPO value clip range"
    )
    parser.add_argument(
        "--lambd", type=float, default=0.95, help="PPO GAE lambd"
    )
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument(
        "--micro_train_batch_size",
        type=int,
        default=4,
        help="batch size per GPU",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        help="Global training batch size",
    )
    parser.add_argument(
        "--normalize_reward",
        action="store_true",
        default=False,
        help="Enable Reward Normazation",
    )
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freezing_actor_steps",
        type=int,
        default=-1,
        help="Used for critic initialization",
    )
    parser.add_argument(
        "--n_samples_per_prompt",
        type=int,
        default=1,
        help="number of responses for each prompt in generation",
    )
    parser.add_argument(
        "--save_value_network",
        action="store_true",
        default=False,
        help="Save critic model",
    )
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument(
        "--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO"
    )
    parser.add_argument(
        "--kl_estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
        help=(
            "In GRPO, k3 is utilized as the loss function, while k2, "
            "when used as the loss, is nearly equivalent to k1."
        ),
    )
    parser.add_argument(
        "--aux_loss_coef", type=float, default=0, help="MoE balancing loss"
    )
    parser.add_argument(
        "--adam_betas",
        type=float,
        nargs=2,
        default=(0.9, 0.95),
        help="Betas for Adam optimizer",
    )
    parser.add_argument(
        "--reward_clip_range",
        type=float,
        nargs=2,
        default=(-10, 10),
        help="Reward clip range",
    )

    # Reinforce
    parser.add_argument(
        "--advantage_estimator",
        type=str,
        choices=[
            "gae",
            "reinforce",
            "rloo",
            "reinforce_baseline",
            "group_norm",
        ],
        default="gae",
        help="Choose advantage estimation method: gae, reinforce, "
        "rloo, reinforce_baseline, group_norm",
    )
    parser.add_argument(
        "--use_kl_loss",
        action="store_true",
        default=False,
        help="whether to use KL loss from GRPO",
    )

    # Context Parallel
    parser.add_argument(
        "--ring_attn_size",
        type=int,
        default=1,
        help="Ring attention group size",
    )
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will "
        "consume more memory.",
    )

    #  Models
    parser.add_argument(
        "--pretrain", type=str, default=None, help="HF model name or path"
    )
    parser.add_argument(
        "--reward_pretrain",
        type=str,
        default=None,
        help="HF model name or path",
    )
    parser.add_argument(
        "--remote_rm_url", type=str, default=None, help="remote RM API (HTTP)"
    )
    parser.add_argument(
        "--critic_pretrain",
        type=str,
        default=None,
        help="HF model name or path",
    )
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument(
        "--ref_reward_offload", action="store_true", default=False
    )

    # Custom dataset
    parser.add_argument(
        "--prompt_data", type=str, default=None, help="HF dataset name or path"
    )
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument(
        "--pretrain_data",
        type=str,
        default=None,
        help="HF dataset name or path",
    )
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--pretrain_split", type=str, default="train")

    parser.add_argument(
        "--input_key", type=str, default="input", help="JSON dataset key"
    )
    parser.add_argument(
        "--label_key", type=str, default=None, help="JSON dataset key"
    )
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="Use HF tokenizer chat template",
    )

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument(
        "--wandb_project", type=str, default="openrlhf_train_ppo"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument(
        "--use_tensorboard",
        type=str,
        default=None,
        help="TensorBoard logging path",
    )

    # performance tuning
    parser.add_argument("--perf", action="store_true", default=False)

    # ModelScope parameters
    parser.add_argument("--use_ms", action="store_true", default=False)

    args = parser.parse_args()

    if args.advantage_estimator not in ["gae"]:
        args.critic_pretrain = None
    elif args.critic_pretrain is None:
        if not args.remote_rm_url:
            args.critic_pretrain = args.reward_pretrain.split(",")[0]
        else:
            args.critic_pretrain = args.pretrain

    if args.advantage_estimator in [
        "rloo",
        "reinforce_baseline",
        "group_norm",
    ]:
        assert (
            args.n_samples_per_prompt > 1
        ), f"{args.advantage_estimator} requires n_samples_per_prompt > 1"

    if args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.input_template and "{}" not in args.input_template:
        logger.warning("{} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        logger.warning(
            "input_template contains \\n chracters instead of "
            "newline. You likely want to pass $'\\n' in Bash or \"`n\" in "
            "PowerShell."
        )

    if args.packing_samples:
        if not args.flash_attn:
            logger.warning(
                "Please --flash_attn to accelerate when "
                "--packing_samples is enabled."
            )
            args.flash_attn = True
        assert (
            args.vllm_num_engines > 0
        ), "Only support `--packing_samples` with vLLM."
        assert (
            not args.pretrain_data
        ), "`--pretrain_data` is not supported with `--packing_samples` yet."

    if args.vllm_enable_sleep and not args.colocate_all_models:
        logger.info(
            "Set args.vllm_enable_sleep to False when "
            "args.colocate_all_models is disabled."
        )
        args.vllm_enable_sleep = False

    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    submit(args)
