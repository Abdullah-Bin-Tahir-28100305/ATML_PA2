from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions.

    rewards, values, mask: [batch, response_steps]. Padding positions must have mask=0.
    The final valid response position bootstraps with zero.
    """
    batch, steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(steps)):
        current_valid = mask[:, t]
        if t + 1 < steps:
            next_valid = mask[:, t + 1]
            next_value = values[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + values
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    rewards = -float(beta_kl) * (policy_logp - ref_logp) * response_mask
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b]
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Return PPO clipped policy loss and diagnostics.

    Manual objective:  L_clip = E_t[ min( rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t ) ].

    DEFECT FIXED: the released starter used `torch.maximum`. The max turns PPO's pessimistic
    lower bound into an optimistic upper bound: for A>0 it keeps the *unclipped* term when the
    ratio exceeds 1+eps (so the update is never limited), and for A<0 it keeps the clipped term
    when the ratio falls below 1-eps, so the trust region no longer constrains the update.
    With `min`, once the ratio leaves the trust region in the direction that improves the
    objective, the clipped (constant) branch is selected and its gradient is zero.
    """
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage

    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(affected, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def clip_diagnostics(ratio, advantage, mask, eps):
    """Diagnostics for the cached-rollout clipping study.

    clip_fraction:   fraction of valid tokens with rho outside [1-eps, 1+eps] (manual definition).
    affected_fraction: fraction of valid tokens where the clipped branch is the one selected by the
        min, i.e. the token's gradient is zeroed by clipping: (A>0 and rho>1+eps) or (A<0 and rho<1-eps).
    """
    outside = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    active = (((advantage > 0) & (ratio > 1.0 + eps)) | ((advantage < 0) & (ratio < 1.0 - eps))).float()
    unclipped = ratio * advantage
    clipped = torch.minimum(unclipped, ratio.clamp(1.0 - eps, 1.0 + eps) * advantage)
    return {
        "clip_fraction": float(masked_mean(outside, mask).item()),
        "affected_fraction": float(masked_mean(active, mask).item()),
        "clipped_surrogate": float(masked_mean(clipped, mask).item()),
        "unclipped_surrogate": float(masked_mean(unclipped, mask).item()),
    }


def value_mse_loss(predicted_values, returns, mask):
    return masked_mean((predicted_values - returns) ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    return ((advantages - mean) / std) * mask
