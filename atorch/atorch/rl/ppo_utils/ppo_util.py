import torch

from atorch.rl.model_utils.model_util import get_tensor_stats, logprobs_of_labels, whiten


def dist_fn(p):
    """
    Creates a categorical distribution parameterized by logits

    Args:
        p: logits, torch.tensor, batch_size*sequence_length*embedding_size
    Return:
        categorical distribution
    """
    return torch.distributions.Categorical(logits=p)


# based on https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
def get_kl_penalty(logits, ref_logits, samples, glm_generation_inputs, attention_mask, kl_controller_value):
    """
    Args:
        logits: logits from actor, torch.tensor, batch_size*sequence_length*embedding_size.
                Logits is compromised of prompts_logits, padding_logits, response_logits and only response logits
                is needed.
        ref_logits: ref logits from ref model, torch.tensor, batch_size*sequence_length*embedding_size
        samples:
        attention: torch.tensor
        kl_controller_value:
    Return:
        logprobs: logprobs for response
        kl_penalty: kl_penalty for response
        mean_kl: average for response
    """
    stats = {}
    logprobs = logprobs_of_labels(logits[:, :-1, :], samples[:, 1:])
    ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], samples[:, 1:])
    prompt_tensors = glm_generation_inputs.input_ids
    start = prompt_tensors.shape[1] - 1
    ends = start + attention_mask[:, start:].sum(1)
    n_samples = prompt_tensors.shape[0]
    log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
    mean_kl = (log_ratio.exp() - 1 - log_ratio).mean()
    kl_penalty = kl_controller_value * -log_ratio

    # to be added to tensorboard
    stats["policy/mean_log_ratio"] = log_ratio.mean().item()
    stats["policy/mean_kl_penalty"] = kl_penalty.mean().item()

    kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]
    logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]
    return logprobs, kl_penalty, mean_kl, stats


# based on https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
def get_rewards(kl_penalty, scores, logprobs):
    """
    get_rewards

    Args:
        kl_penalty: a list of tensors, batch_size*sequence_length
        scores: torch.tensor, scalar calculated by reward model for every str_sample
        logprobs: torch.tensor, batch_size*sequence_length*embedding_size
    Return:
        categorical distribution
    """

    n_samples = scores.shape[0]
    all_rewards = []
    for sample_idx in range(n_samples):
        if len(kl_penalty[sample_idx]) == 0 or len(logprobs[sample_idx]) == 0:
            continue
        rewards = kl_penalty[sample_idx]
        rewards[-1] += scores[sample_idx]
        all_rewards.append(rewards)
    return all_rewards


# based on https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
def loss(logprobs, values, old_logprobs, old_values, advantages, returns, mask, logits, config=None):
    """PPO objective function.
    References:
    - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
    """
    clip_ratio = config.clip_ratio
    cliprange_value = config.cliprange_value
    cliprange = config.cliprange
    vf_coef = config.vf_coef
    ent_coef = config.ent_coef
    values_clipped = torch.clamp(
        values,
        old_values - cliprange_value,
        old_values + cliprange_value,
    )
    n = mask.sum()

    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
    vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
    if clip_ratio:
        ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    with torch.no_grad():
        approx_kl = torch.mean((ratio - 1) - log_ratio)  # noqa: F841

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(
        ratio,
        1.0 - cliprange,
        1.0 + cliprange,
    )
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
    pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

    # compared with trlx loss function, entroy loss is added
    dist = dist_fn(logits)
    ent_loss = torch.sum(dist.entropy() * mask) / n

    loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

    stats = dict(
        losses=dict(
            total_loss=loss.item(),
            policy_loss=pg_loss.item(),
            value_loss=vf_loss.item(),
            entropy_loss=ent_loss.item(),
        ),
        values=dict(
            get_tensor_stats(values, mask, n),
            values_error=(torch.sum(((values - returns) * mask) ** 2) / n).item(),
            clipfrac=vf_clipfrac.item(),
        ),
        old_values=get_tensor_stats(old_values, mask, n),
        returns=get_tensor_stats(returns, mask, n),
        policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item()),
        ratio=((ratio * mask).sum().item() / n).item(),
        padding_percentage=(n / mask.numel()).item(),
    )
    return loss, stats


# based on https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py
def get_advantages_and_returns(values, rewards, response_length, use_whitening=True, config=None):
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Args:
        values: Tensor of shape (batch_size, response_size)
        rewards: Tensor of shape (batch_size, response_size)
        response_length: Length of the response sequence
        use_whitening: Whether to use whitening (ie. normalize advantages) or not
    """
    gamma = config.gamma
    lam = config.lam
    lastgaelam = 0

    advantages_reversed = []
    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    if use_whitening:
        advantages = whiten(advantages)
    return advantages.detach(), returns
