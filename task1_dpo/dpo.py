from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Manual objective:
        L = -E[ log sigmoid( beta * ( log pi(y+|x)/pi_ref(y+|x) - log pi(y-|x)/pi_ref(y-|x) ) ) ]

    Rearranging the two log-ratios gives
        beta * ( [log pi(y+) - log pi(y-)] - [log pi_ref(y+) - log pi_ref(y-)] )
      = beta * ( policy_margin - ref_margin ).

    DEFECT FIXED: the released starter used `policy_margin + ref_margin`, which adds the
    reference log-ratio instead of subtracting it. That objective rewards the policy for
    whatever the reference already prefers and no longer measures change relative to the
    reference policy (at initialisation, policy == reference, so the logit should be exactly 0
    and the loss log(2); the buggy version gives 2*beta*ref_margin instead).
    """

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # m_theta in the manual's preference-accuracy definition.
    preference_margin = policy_margin - ref_margin
    logits = beta * preference_margin

    loss = -F.logsigmoid(logits).mean()

    # Implicit DPO rewards, useful for logging: r = beta * log(pi / pi_ref).
    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "preference_margin_mean": preference_margin.detach().mean(),
        "chosen_reward_mean": chosen_reward.detach().mean(),
        "rejected_reward_mean": rejected_reward.detach().mean(),
        "preference_accuracy": (preference_margin > 0).float().mean().detach(),
    }
