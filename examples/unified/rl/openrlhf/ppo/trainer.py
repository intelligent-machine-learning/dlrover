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

import os
import time
from concurrent import futures
from contextlib import contextmanager
from datetime import timedelta

import ray
import torch
from omegaconf import DictConfig
from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.utils import get_strategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer
from tqdm import tqdm

from . import remote_call

logger = init_logger(__name__)


# Modified from openrlhf.trainer.ppo_trainer.py
class BasePPOTrainer:
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(self, config: DictConfig):
        strategy = get_strategy(config)
        self.strategy = strategy
        self.args = config

        self.freezing_actor_steps = config.get("freezing_actor_steps", -1)
        # get eval and save steps
        if config.eval_steps == -1:
            config.eval_steps = float("inf")  # do not evaluate
        if config.save_steps == -1:
            config.save_steps = float("inf")  # do not save ckpt

        self.tokenizer = get_tokenizer(
            self.args.pretrain,
            None,
            "left",
            strategy,
            use_fast=not self.args.disable_fast_tokenizer,
        )

        from .utils.experience_maker import (
            RemoteExperienceMaker,
            SamplesGenerator,
        )

        if config.kl_target:
            self.kl_ctl = AdaptiveKLController(
                config.init_kl_coef, config.kl_target, config.kl_horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.init_kl_coef)

        self.samples_generator = SamplesGenerator(
            config,
            self.tokenizer,
            self.args.prompt_max_len,
        )
        self.generate_kwargs = dict(
            do_sample=True,
            max_new_tokens=config.generate_max_len,
            max_length=config.max_len,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        self.experience_maker = RemoteExperienceMaker(
            self.kl_ctl,
            self.args,
            self.tokenizer,
        )

    def init_workers(self):
        strategy = self.strategy
        args = self.args

        # TODO vllm init itself
        futures.wait(
            [
                remote_call.reference_init(strategy, args.pretrain),
                remote_call.reward_init(
                    strategy, args.reward_pretrain.split(",")[0]
                ),
                remote_call.actor_init(
                    strategy, args.pretrain, max_steps=self.max_steps
                ),
                remote_call.critic_init(
                    strategy,
                    args.critic_pretrain,
                    max_steps=self.max_steps,
                ),
            ]
        )
        logger.info("Done all model init")

    def _init_wandb(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None
        if self.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.args.use_wandb)
            wandb.init(
                entity=self.args.wandb_org,
                project=self.args.wandb_project,
                group=self.args.wandb_group,
                name=self.args.wandb_run_name,
                config=self.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric(
                "train/*", step_metric="train/global_step", step_sync=True
            )
            wandb.define_metric("eval/epoch")
            wandb.define_metric(
                "eval/*", step_metric="eval/epoch", step_sync=True
            )
            self.generated_samples_table = wandb.Table(
                columns=["global_step", "text", "reward"]
            )

        # Initialize TensorBoard writer if wandb is not available
        if self.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(
                self.args.use_tensorboard,
                self.args.wandb_run_name,
            )
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
    ) -> None:
        args = self.args

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(
                self.actor_model_group.async_run_method(
                    method_name="get_checkpoint_states"
                )
            )[0]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            self._broadcast_to_vllm()
        else:
            checkpoint_states = {
                "global_step": 0,
                "episode": 0,
                "data_loader_state_dict": {},
            }

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)

        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
            )

            for _, rand_prompts, labels in self.prompts_dataloader:
                rollout_samples = self.samples_generator.generate_samples(
                    rand_prompts,
                    labels,
                    **self.generate_kwargs,
                )
                pbar.update()

                experiences = self.experience_maker.make_experience_batch(
                    rollout_samples
                )
                sample0 = self.tokenizer.batch_decode(
                    experiences[0].sequences[0].unsqueeze(0),
                    skip_special_tokens=True,
                )
                print(sample0)

                # balance experiences across dp
                if args.use_dynamic_batch:
                    experiences = balance_experiences(experiences, args)

                # append experiences to remote actor and critic
                futures.wait(
                    [
                        remote_call.actor_append_experience(experiences),
                        remote_call.critic_append_experience(experiences),
                    ]
                )
                status = self.ppo_train(steps)

                if "kl" in status:
                    self.kl_ctl.update(
                        status["kl"],
                        args.rollout_batch_size * args.n_samples_per_prompt,
                    )

                # Add generated samples to status dictionary
                logger.info(f"✨ Global step {steps}: {status}")
                status["generated_samples"] = [
                    sample0[0],
                    experiences[0].info["reward"][0],
                ]

                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(
                    args, steps, pbar, status, client_states
                )

                steps = steps + 1

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()

    def ppo_train(self, global_steps):
        status = {}

        critic_status_ref = remote_call.critic_train()

        # actor model training
        if global_steps > self.freezing_actor_steps:
            actor_status_ref = remote_call.actor_train(
                kl_ctl=self.kl_ctl.value
            )
            status.update(actor_status_ref.result()[0])

            # 4. broadcast weights to vllm engines
            self._broadcast_to_vllm()

        # 5. wait remote critic model training done
        status.update(critic_status_ref.result()[0])
        return status

    @contextmanager
    def vllm_wakeup(self):
        """Context manager to wake up vLLM engines."""
        if self.args.vllm_enable_sleep:
            remote_call.vllm_wakeup()
        try:
            yield
        finally:
            if self.args.vllm_enable_sleep:
                remote_call.vllm_sleep()

    def _broadcast_to_vllm(self):
        with self.vllm_wakeup():
            remote_call.actor_sync_to_vllm()

    def save_logs_and_checkpoints(
        self, args, global_step, step_bar, logs_dict={}, client_states={}
    ):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                if "generated_samples" in logs_dict:
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns,
                        data=self.generated_samples_table.data,
                    )
                    new_table.add_data(
                        global_step, *logs_dict.pop("generated_samples")
                    )
                    self.generated_samples_table = new_table
                    self._wandb.log({"train/generated_samples": new_table})
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples":
                        # Record generated samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = (
                            f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        )
                        self._tensorboard.add_text(
                            "train/generated_samples",
                            formatted_text,
                            global_step,
                        )
                    else:
                        self._tensorboard.add_scalar(
                            f"train/{k}", v, global_step
                        )

        # TODO: Add evaluation mechanism for PPO
        if (
            global_step % args.eval_steps == 0
            and self.eval_dataloader
            and len(self.eval_dataloader) > 0
        ):
            self.evaluate(
                self.eval_dataloader,
                global_step,
                args.eval_temperature,
                args.eval_n_samples_per_prompt,
            )
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            futures.wait(
                [
                    remote_call.actor_save_model(
                        os.path.join(args.ckpt_path, "_actor"),
                        tag=tag,
                        ext_states=client_states,
                    ),
                    remote_call.critic_save_model(
                        os.path.join(args.ckpt_path, "_critic"), tag=tag
                    ),
                ]
            )

    def evaluate(
        self,
        eval_dataloader,
        global_step,
        temperature=0.6,
        n_samples_per_prompt=1,
    ):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(
            f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        with self.vllm_wakeup(), torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources

            for datasources, prompts, labels in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                # Create mapping for each prompt to its corresponding data source
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            # Generate samples and calculate rewards
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            samples_list = self.samples_generator.generate_samples(
                all_prompts,
                all_labels,
                **generate_kwargs,
            )

            # duplicate prompts and labels for each sample
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])

            # Get rewards from samples, such as agent rewards or remote reward models
            rewards_list = []
            for samples in samples_list:
                rewards_list.append(samples.rewards)
            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            rewards = torch.tensor(rewards_list).reshape(
                -1, n_samples_per_prompt
            )

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

            # Process rewards in chunks of n_samples_per_prompt
            num_prompts = len(all_prompts) // n_samples_per_prompt
            for i in range(num_prompts):
                # Get the original prompt (first one in the chunk)
                original_prompt = all_prompts[i * n_samples_per_prompt]
                datasource = prompt_to_datasource[
                    original_prompt
                ]  # Get corresponding data source using the mapping

                if datasource not in global_metrics:
                    global_metrics[datasource] = {
                        f"pass{n_samples_per_prompt}": 0,
                        "pass1": 0,
                        "count": 0,
                    }

                # Get rewards for this chunk
                chunk_rewards = rewards[i]

                # Calculate pass@k and pass@1
                if n_samples_per_prompt > 1:
                    global_metrics[datasource][
                        f"pass{n_samples_per_prompt}"
                    ] += chunk_rewards.max().float().item()
                global_metrics[datasource]["pass1"] += (
                    chunk_rewards.mean().float().item()
                )
                global_metrics[datasource]["count"] += 1

            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                    metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                )
                logs[f"eval_{datasource}_pass1"] = (
                    metrics["pass1"] / metrics["count"]
                )

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {
                    "eval/%s" % k: v
                    for k, v in {**logs, "global_step": global_step}.items()
                }
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(
            f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}"
        )

    def prepare_datasets(self):
        args = self.args
        strategy = self.strategy

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            dataset_split=args.prompt_split,
        )

        # Create train dataset
        train_data = train_data.select(
            range(min(args.max_samples, len(train_data)))
        )
        prompts_dataset = PromptDataset(
            train_data,
            self.tokenizer,
            strategy,
            input_template=args.input_template,
        )
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.vllm_generate_batch_size,
            True,
            True,
        )

        # Create eval dataset if eval data exists
        if getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
                dataset_split=args.eval_split,
            )
            eval_data = eval_data.select(
                range(min(args.max_samples, len(eval_data)))
            )
            eval_dataset = PromptDataset(
                eval_data,
                self.tokenizer,
                strategy,
                input_template=args.input_template,
            )
            eval_dataloader = strategy.setup_dataloader(
                eval_dataset, 1, True, False
            )
        else:
            eval_dataloader = None

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_steps = (
            len(prompts_dataset)
            * args.n_samples_per_prompt
            // args.train_batch_size
            * args.num_episodes
            * args.max_epochs
        )
