"""Numerical checks of the three corrected objectives (no models needed).

python -m scripts.check_objectives
"""

import math

import torch

from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import clip_diagnostics, ppo_policy_loss
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss


def main():
    # DPO: policy == reference -> logit 0 -> loss log 2; raising the chosen log-prob lowers the loss.
    pc, pr, rc, rr = torch.tensor([-10.0]), torch.tensor([-12.0]), torch.tensor([-10.0]), torch.tensor([-12.0])
    l0, _ = dpo_loss(pc, pr, rc, rr, 0.1)
    l1, d1 = dpo_loss(pc + 1, pr, rc, rr, 0.1)
    assert abs(l0.item() - math.log(2)) < 1e-6 and l1.item() < l0.item() and d1["preference_accuracy"].item() == 1.0
    print(f"DPO  init loss {l0.item():.5f} (log 2 = {math.log(2):.5f}); chosen +1 nat -> {l1.item():.5f}")

    # PPO: outside the trust region in the improving direction -> clipped value, zero gradient.
    for adv, rho, expect in [(1.0, 1.5, 1.2), (-1.0, 0.5, -0.8)]:
        new = torch.tensor([[math.log(rho)]], requires_grad=True)
        loss, _, _ = ppo_policy_loss(new, torch.zeros(1, 1), torch.full((1, 1), adv), torch.ones(1, 1), 0.2)
        loss.backward()
        assert abs(-loss.item() - expect) < 1e-5 and new.grad.abs().item() < 1e-8
        print(f"PPO  A={adv:+.0f} rho={rho}: objective {-loss.item():.3f} (expect {expect}), grad {new.grad.item():.1e}")
    print("PPO  diagnostics:", clip_diagnostics(torch.tensor([[1.5, 0.5, 1.0, 1.5]]), torch.tensor([[1.0, 1.0, 1.0, -1.0]]), torch.ones(1, 4), 0.2))

    # GRPO: per-group statistics; a constant-reward group gets zero advantage.
    r = torch.tensor([10.0, 10.0, 10.0, 10.0, 0.0, 1.0, 2.0, 3.0])
    adv = group_relative_advantages(r, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
    assert torch.allclose(adv[:4], torch.zeros(4)) and abs(adv[4:].mean().item()) < 1e-6
    print("GRPO advantages:", [round(x, 3) for x in adv.tolist()])

    # grpo (1/T_k) vs dr_grpo (1/L_max): per-token gradient on a short vs a long completion.
    T = 6
    mask = torch.zeros(2, T)
    mask[0, :2] = 1
    mask[1, :] = 1
    for lt in ["grpo", "dr_grpo"]:
        new = torch.zeros(2, T, requires_grad=True)
        loss, _ = grpo_policy_loss(new, new.detach(), torch.ones(2), mask, new.detach(), 0.2, 0.0, loss_type=lt, max_completion_length=T)
        loss.backward()
        print(f"GRPO {lt:8s} per-token grad short={new.grad[0, 0].item():.4f} long={new.grad[1, 0].item():.4f}")
    print("All objective checks passed.")


if __name__ == "__main__":
    main()
