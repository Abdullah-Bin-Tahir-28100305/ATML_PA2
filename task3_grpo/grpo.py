from __future__ import annotations

import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


ZERO_STD_TOL = 1e-6  # numerical tolerance for an "uninformative" (zero-std) group


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion.

    Manual definition, per prompt group:  A_k = (r_k - mu_r) / (sigma_r + eps),
    with mu_r and sigma_r computed over the K completions of the *same prompt*.

    DEFECT FIXED: the released starter normalised with the mean/std of the whole batch. When a
    batch contains several prompts, that compares completions of different prompts with each
    other: an easy prompt's completions all get positive advantages and a hard prompt's all get
    negative ones, even if every group is internally uninformative. Statistics are now computed
    within each group, so a group whose rewards are all equal gets advantage 0 for every member.
    """
    rewards = rewards.float()
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    adv = torch.zeros_like(rewards)
    for gid in torch.unique(group_ids):
        idx = group_ids == gid
        r = rewards[idx]
        mean = r.mean()
        std = r.std(unbiased=False)
        adv[idx] = (r - mean) / (std + eps)
    return adv


def group_reward_stats(rewards: torch.Tensor, group_ids: torch.Tensor, tol: float = ZERO_STD_TOL):
    """Per-group reward std plus the uninformative-group fraction (std <= tol)."""
    group_ids = torch.as_tensor(group_ids, device=rewards.device)
    stds = []
    for gid in torch.unique(group_ids):
        stds.append(float(rewards[group_ids == gid].float().std(unbiased=False).item()))
    stds_t = torch.tensor(stds)
    return {
        "group_reward_std_mean": float(stds_t.mean().item()),
        "frac_zero_std_groups": float((stds_t <= tol).float().mean().item()),
        "group_reward_stds": stds,
    }


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions.

    `token_mask` may be all-zero for a completion that was deliberately masked because it hit the
    maximum generation length.
    """
    ratio = torch.exp(new_logp - old_logp)
    adv = seq_adv[:, None]
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length.
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = ref_logp - new_logp
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep
