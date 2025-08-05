import time
from abc import ABC
from copy import deepcopy
from datetime import timedelta
from typing import List, Tuple

import ray
import torch
from openrlhf.models.utils import (
    compute_approx_kl,
    compute_reward,
    masked_mean,
)
from openrlhf.trainer.ppo_utils.experience_maker import (
    Experience,
    logger,
    update_samples_with_rewards,
)
from openrlhf.utils.seqlen_balancing import (
    get_minimum_num_micro_batch_size,
    get_seqlen_balanced_partitions,
)
from openrlhf.utils.utils import remove_pad_token

from examples.unified.rl.openrlhf.ppo import remote_call


class SamplesGenerator:
    def __init__(self, args, tokenizer, prompt_max_len):
        self.args = args
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len

    @torch.no_grad()
    def generate_samples(
        self, all_prompts: List[str], all_labels, **generate_kwargs
    ) -> List[Experience]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        with remote_call.vllm_running(self.args.vllm_enable_sleep):
            rollout_samples = self._generate_vllm(
                all_prompts, all_labels, **generate_kwargs
            )

        return rollout_samples

        # tokenizer

    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _generate_vllm(
        self, all_prompts: List[str], all_labels, **kwargs
    ) -> List[Experience]:
        """Generate samples using vLLM engine.

        Args:
            all_prompts: List of prompts to generate from
            all_labels: List of labels corresponding to prompts
            **kwargs: Additional arguments for generation

        Returns:
            List of Experience objects containing generated samples
        """
        from vllm import SamplingParams

        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )
        max_response_length = kwargs.get("max_new_tokens", 1024)
        truncate_length = self.prompt_max_len + max_response_length

        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = kwargs.pop(
            "n_samples_per_prompt", self.args.n_samples_per_prompt
        )
        all_prompts = sum(
            [[prompt] * n_samples_per_prompt for prompt in all_prompts], []
        )
        all_labels = sum(
            [[label] * n_samples_per_prompt for label in all_labels], []
        )
        all_prompt_token_ids = self.tokenize_fn(
            all_prompts, self.prompt_max_len, padding=False
        )["input_ids"]

        all_outputs = remote_call.vllm_generate(
            all_prompt_token_ids, sampling_params
        )

        pad_token_id, eos_token_id = (
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        )

        # Process outputs into Experience objects
        samples_list = []
        for i in range(len(all_outputs)):
            output = all_outputs[i]
            prompt = all_prompts[i]
            label = all_labels[i]

            # Concatenate prompt and output tokens
            input_ids = list(output.prompt_token_ids) + list(
                output.outputs[0].token_ids
            )
            if output.outputs[0].token_ids[-1] != eos_token_id:
                input_ids.append(eos_token_id)
            attention_mask = [1] * len(input_ids)

            sequences = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            # Create action mask based on output token positions
            action_mask = torch.zeros_like(attention_mask)
            response_length = len(output.outputs[0].token_ids) + int(
                output.outputs[0].token_ids[-1] != eos_token_id
            )
            action_mask[
                len(output.prompt_token_ids) : len(output.prompt_token_ids)
                + response_length
            ] = 1

            sequences = sequences[:truncate_length].to("cpu")
            attention_mask = attention_mask[:truncate_length].to("cpu")
            action_mask = action_mask[1:truncate_length].to("cpu")
            total_length = attention_mask.float().sum()
            is_clipped = response_length >= max_response_length

            info = {
                "response_length": torch.tensor([response_length]),
                "total_length": torch.tensor([total_length]),
                "response_clip_ratio": torch.tensor([is_clipped]),
            }

            rollout_samples = Experience(
                sequences=sequences.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                action_mask=action_mask.unsqueeze(0),
                prompts=[prompt],
                labels=[label],
                info=info,
            )
            samples_list.append(rollout_samples)

        # Get rewards from remote reward models if needed
        # This is required by dynamic sampling
        remote_reward_model = kwargs.get("remote_reward_model", None)
        if remote_reward_model:
            all_queries = sum(
                [
                    self.tokenizer.batch_decode(
                        remove_pad_token(s.sequences, s.attention_mask),
                        skip_special_tokens=False,
                    )
                    for s in samples_list
                ],
                [],
            )
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])

            # Get rewards info from remote model
            rewards_info = ray.get(
                remote_reward_model.get_rewards.remote(
                    all_queries, all_prompts, all_labels
                )
            )
            # Process rewards and scores
            update_samples_with_rewards(rewards_info, samples_list)

        return samples_list


class RemoteExperienceMaker(ABC):
    def __init__(
        self,
        kl_controller,
        args,
        tokenizer,
        **kwargs,
    ):
        super().__init__()

        self.kl_ctl = kl_controller
        self.advantage_estimator = args.advantage_estimator
        self.args = args

        self.tokenizer = tokenizer

    def split_rollout_samples(self, rollout_samples):
        for i, sample in enumerate(rollout_samples):
            sample.index = [i]

        samples_list = []
        if self.args.use_dynamic_batch:
            total_lengths = [
                int(s.info["total_length"].item()) for s in rollout_samples
            ]
            effective_actor_num = (
                self.args.actor_num_nodes
                * self.args.actor_num_gpus_per_node
                // self.args.ring_attn_size
                // self.args.ds_tensor_parallel_size
            )
            minimum_batch_num = get_minimum_num_micro_batch_size(
                total_lengths,
                self.args.rollout_max_tokens_per_gpu,
                self.args.ring_attn_size,
                self.args.ds_tensor_parallel_size,
            )
            minimum_batch_num = (
                minimum_batch_num // effective_actor_num * effective_actor_num
            )
            num_batch = max(minimum_batch_num, effective_actor_num)
            batch_indexes = get_seqlen_balanced_partitions(
                total_lengths, num_batch, False
            )
            for micro_index in batch_indexes:
                micro_batch = [rollout_samples[idx] for idx in micro_index]
                concat_samples = Experience.concat_experiences(
                    micro_batch, self.tokenizer.pad_token_id
                )
                samples_list.append(concat_samples)
        else:
            batch_size = self.args.micro_rollout_batch_size
            for i in range(0, len(rollout_samples), batch_size):
                concat_samples = Experience.concat_experiences(
                    rollout_samples[i : i + batch_size],
                    self.tokenizer.pad_token_id,
                )
                samples_list.append(concat_samples)
        return samples_list

    @torch.no_grad()
    def make_experience_batch(self, rollout_samples) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # Each batch of samples will be scheduled to a effective Ray Actor (i.e, a DP rank)
        samples_list = self.split_rollout_samples(rollout_samples)

        # Make experiences (models forward: logprobs, values, rewards, and kl divergence)
        experiences = self.make_experience(samples_list)

        # Process experiences (reward shaping, etc.)
        experiences = self.compute_advantages_and_returns(experiences)
        return experiences

    @torch.no_grad()
    def make_experience(
        self, samples_list: List[Experience]
    ) -> List[Experience]:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        start_time = time.time()
        logger.info(
            f"🚀 Starting experience making with {sum([len(s.sequences) for s in samples_list])} samples"
        )

        args = self.args
        device = "cpu"

        # Extract all information from samples in one pass
        # Convert samples into lists of tensors and metadata for batch processing
        sequences_list = [s.sequences for s in samples_list]
        attention_mask_list = [s.attention_mask for s in samples_list]
        action_mask_list = [s.action_mask for s in samples_list]

        # The rewards are already filled in the samples_list, such as the agent's environment rewards
        if samples_list[0].rewards is not None:
            pass
        else:
            r_refs = remote_call.reward_forward(
                sequences=sequences_list,
                attention_mask=attention_mask_list,
            )

        # Batch call actor model
        action_log_probs_ref = remote_call.actor_forward(
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
        )

        # Batch call critic model
        value_ref = remote_call.critic_forward(
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
        )

        # Batch call initial model
        base_action_log_probs_ref = remote_call.reference_forward(
            sequences=sequences_list,
            action_mask=action_mask_list,
            attention_mask=attention_mask_list,
        )

        # Wait for all remote calls to complete and flatten the results
        # Note: the results duplicated ring_attn_size * ds_tensor_parallel_size times
        # This is because the actors in ring group and tp group will return the same output
        duplicate_factor = args.ring_attn_size * args.ds_tensor_parallel_size
        action_log_probs_list = sum(
            action_log_probs_ref.result()[::duplicate_factor], []
        )
        base_action_log_probs_list = sum(
            base_action_log_probs_ref.result()[::duplicate_factor], []
        )
        value_list = sum(value_ref.result()[::duplicate_factor], [])

        # Process rewards based on source
        if samples_list[0].rewards is not None:
            pass
        else:
            # Reward Model
            rewards_list = sum(r_refs.result()[::duplicate_factor], [])
            for i, samples in enumerate(samples_list):
                samples.rewards = rewards_list[i]
                samples.info["reward"] = rewards_list[i]

        assert (
            len(samples_list)
            == len(action_log_probs_list)
            == len(base_action_log_probs_list)
            == len(value_list)
        ), (
            f"len(samples_list): {len(samples_list)}, len(action_log_probs_list): {len(action_log_probs_list)}, len(base_action_log_probs_list): {len(base_action_log_probs_list)}, len(value_list): {len(value_list)}"
        )

        # Process results for each sample
        for i, (
            samples,
            action_log_probs,
            base_action_log_probs,
            value,
        ) in enumerate(
            zip(
                samples_list,
                action_log_probs_list,
                base_action_log_probs_list,
                value_list,
            )
        ):
            if base_action_log_probs is not None and (not args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    kl_estimator=self.args.kl_estimator,
                )
            else:
                kl = torch.zeros_like(
                    action_log_probs,
                    dtype=action_log_probs.dtype,
                    device=device,
                )
            kl_mean = masked_mean(kl, samples.action_mask, dim=-1)

            if not args.use_kl_loss:
                base_action_log_probs = None

            # Update experience with new information
            samples.action_log_probs = action_log_probs
            samples.base_action_log_probs = base_action_log_probs
            samples.values = value
            samples.kl = kl
            samples.info["kl"] = kl_mean

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Experience making completed in {time_str}")
        return samples_list

    @torch.no_grad()
    def compute_advantages_and_returns(
        self, experiences: List[Experience], **kwargs
    ) -> List[Experience]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.
        Example, use_dynamic_batch
            >>> rewards: [0, 1, 0.5, 1], indices: [1, 2, 0, 3], n_samples_per_prompt: 2
            >>> sorted rewards: [0,5, 0, 1, 1], reward shaping: [0.25, 0.25, 1, 1]
            >>> map back: [0.25, 1, 0.25, 1]
        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.args

        # DAPO reward shaping with optional overlong penalty - Apply BEFORE dynamic indices processing
        if args.overlong_buffer_len is not None:
            assert args.generate_max_len >= args.overlong_buffer_len, (
                "generate_max_len must be larger than overlong_buffer_len"
            )
            overlong_buffer_len = args.overlong_buffer_len
            expected_len = args.generate_max_len - overlong_buffer_len
            overlong_penalty_factor = args.overlong_penalty_factor

            # Apply penalty to each experience's rewards based on response length
            for experience in experiences:
                response_lengths = experience.info["response_length"]
                batch_size = len(response_lengths)
                for j in range(batch_size):
                    valid_response_length = response_lengths[j].item()
                    # Cap the exceed_len to overlong_buffer_len to prevent excessive penalty
                    exceed_len = min(
                        valid_response_length - expected_len,
                        overlong_buffer_len,
                    )
                    if exceed_len > 0:
                        overlong_penalty = (
                            -exceed_len
                            / overlong_buffer_len
                            * overlong_penalty_factor
                        )
                        # Apply penalty to the j-th reward in this experience
                        experience.rewards[j] += overlong_penalty

        # get rewards from experiences
        exp_len = [len(experience.index) for experience in experiences]
        # indices is an identity mapping when not using dynamic batch; otherwise, it maps back to the original indices after rearange samples
        indices = torch.tensor(
            sum([experience.index for experience in experiences], [])
        )
        raw_rewards = torch.cat(
            [experience.rewards for experience in experiences], dim=0
        )
        rewards = torch.empty_like(raw_rewards)
        rewards[indices] = raw_rewards  # sorted

        rewards = rewards.reshape(-1, args.n_samples_per_prompt)

        # log group reward std
        if args.n_samples_per_prompt > 1:
            group_reward_stds = (
                rewards.std(-1, keepdim=True)
                .repeat(1, args.n_samples_per_prompt)
                .reshape(-1)[indices]
                .split(exp_len)
            )
            for experience, group_reward_std in zip(
                experiences, group_reward_stds
            ):
                experience.info["group_reward_std"] = group_reward_std

        # reward shaping
        if args.advantage_estimator == "rloo":
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (
                args.n_samples_per_prompt - 1
            )
            rewards = rewards - baseline
        elif args.advantage_estimator in ["reinforce_baseline", "dr_grpo"]:
            # REINFORCE++-baseline and Dr. GRPO removed the `/std` in GRPO as `/ std` is not needed in RL variance reduction theory.
            # And `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = rewards - rewards.mean(-1, keepdim=True)
        elif args.advantage_estimator == "group_norm":
            rewards = (rewards - rewards.mean(-1, keepdim=True)) / (
                rewards.std(-1, keepdim=True) + 1e-9
            )

        rewards = rewards.reshape(-1)[indices].split(exp_len)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = (
                    self.get_advantages_and_returns(
                        experience.values,
                        reward,
                        experience.action_mask,
                        args.gamma,
                        args.lambd,
                    )
                )
            elif self.advantage_estimator in [
                "reinforce",
                "rloo",
                "reinforce_baseline",
                "group_norm",
                "dr_grpo",
            ]:
                if args.gamma != 1.0 and self.advantage_estimator in [
                    "rloo",
                    "reinforce_baseline",
                    "group_norm",
                    "dr_grpo",
                ]:
                    logger.warning(
                        "gamma is set to 1.0 for rloo, reinforce_baseline, and group_norm"
                    )
                    args.gamma = 1.0

                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    args.gamma,
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(
                    f"Unkown advantage_estimator {self.advantage_estimator}"
                )

            # calculate the return info.
            return_sums = reward.sum(dim=-1)
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None

        # Normalize advantages across all experiences for GAE, REINFORCE, and REINFORCE-baseline
        if self.args.advantage_estimator in [
            "gae",
            "reinforce",
            "reinforce_baseline",
        ]:
            all_advantages = []
            all_action_masks = []
            for exp in experiences:
                all_advantages.append(exp.advantages.flatten())
                all_action_masks.append(exp.action_mask.flatten())

            advantages_vector = torch.cat(all_advantages, dim=0).float()
            action_masks_vector = torch.cat(all_action_masks, dim=0)
            num_actions = action_masks_vector.sum()

            # mean
            mean = (
                advantages_vector * action_masks_vector
            ).sum() / num_actions
            # std
            if not self.args.no_advantage_std_norm:
                var = (
                    (advantages_vector - mean).pow(2) * action_masks_vector
                ).sum() / num_actions
                rstd = var.clamp(min=1e-8).rsqrt()
            else:
                rstd = 1

            # Apply normalization to each experience
            for exp in experiences:
                exp.advantages = (exp.advantages - mean) * rstd

        return experiences

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns
