"""Task 2 Step 1: PPO continuation from the supplied midpoint policy + critic.

Per update (configs/ppo.yaml):
  1. take the next `prompts_per_update` prompts from a seed-fixed order of the training pool;
  2. sample responses from the current policy (cap `max_response_length`);
  3. without gradients: old-policy log-probs, frozen-reference log-probs (adapter disabled =
     the untouched Qwen2.5-1.5B-Instruct reference), critic values, learned reward from the
     fixed reward model; responses without EOS receive `missing_eos_penalty`;
  4. KL-shaped token rewards r_t = r_task*1[t=T] - beta_KL (log pi_old - log pi_ref);
  5. GAE(gamma, lambda) advantages and returns; advantages are whitened over valid tokens;
  6. `ppo_epochs` passes: clipped policy loss (task2_ppo.ppo, corrected) and critic MSE loss,
     each with its own optimiser and gradient-norm clipping.
Logged per update: learned reward, KL to reference, policy/value loss, entropy, gradient norms,
clip fraction, response length, critic explained variance, wall-clock and peak VRAM.

Every fork (clipping / KL studies) calls `run_ppo` with the same seed, so all branches start from
the identical supplied policy/value state and see the same prompt sequence.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed
from common.metrics import masked_mean, sampled_kl, sample_entropy
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from common.training_utils import (
    Timer,
    disable_dropout,
    finite_or_none,
    gpu_name,
    grads_are_finite,
    peak_vram_gib,
    reset_peak_vram,
    rl_prompt_schedule,
    token_entropy_from_logits,
    upcast_trainable_params,
)
from common.policy_eval import exact_token_kl
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    # No dropout: rollout, reference and training passes must score tokens identically.
    disable_dropout(policy)
    disable_dropout(value_model)
    upcast_trainable_params(policy)
    upcast_trainable_params(value_model)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def explained_variance(values, returns, mask):
    v = values[mask.bool()].float()
    r = returns[mask.bool()].float()
    if r.numel() < 2 or float(r.var()) == 0.0:
        return None
    return float(1.0 - (r - v).var() / r.var())


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = bundle["tokenizer"]
    policy, value_model = bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    pol_opt, val_opt = bundle["policy_optimizer"], bundle["value_optimizer"]
    pol_params, val_params = trainable_parameters(policy), trainable_parameters(value_model)

    seed = int(cfg["seed"])
    n_updates = int(cfg["updates"])
    P = int(cfg["prompts_per_update"])
    eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    epochs = int(cfg["ppo_epochs"])
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    value_coef = float(cfg.get("value_coef", 0.5))
    eos_penalty = float(cfg.get("missing_eos_penalty", 0.0))
    gen_cfg = cfg["generation"]

    schedule, dropped = rl_prompt_schedule(bundle["prompt_rows"], tokenizer, int(cfg["max_prompt_length"]), seed)
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / "train_logs" / f"{run_name}.jsonl"
    roll_path = results_dir / "rollouts" / f"{run_name}.jsonl"
    for p in (log_path, roll_path):
        if p.exists():
            p.unlink()

    print(f"[PPO:{run_name}] updates={n_updates} eps={eps} beta_kl={beta_kl} prompts/update={P} "
          f"epochs={epochs} usable_prompts={len(schedule)} (dropped {len(dropped)} over max_prompt_length)", flush=True)

    timer = Timer()
    reset_peak_vram()
    policy.train()
    value_model.train()
    used_prompt_ids = []

    for u in range(n_updates):
        rows = [schedule[(u * P + j) % len(schedule)] for j in range(P)]
        msgs = [prompt_messages(r) for r in rows]
        used_prompt_ids += [str(r.get("prompt_id")) for r in rows]

        # ---- 1-2. rollout -------------------------------------------------------------
        torch.manual_seed(seed * 1000 + u)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 1000 + u)
        gen = batch_generate(
            policy, tokenizer, msgs,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["max_response_length"]),
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            do_sample=True,
        )
        width = gen["prompt_width"]
        r_len = max(int(gen["response_mask"].sum(-1).max().item()), 1)
        # batch_generate runs under torch.inference_mode; clone so the tensors can enter autograd.
        seq = gen["sequences"][:, : width + r_len].clone()
        attn = gen["attention_mask"][:, : width + r_len].clone()
        resp = gen["response_ids"][:, :r_len].clone()
        mask = gen["response_mask"][:, :r_len].float().clone()

        # ---- 3. old / reference log-probs, values, learned reward ------------------------
        with torch.no_grad():
            old_logp, logits = response_token_logprobs(policy, seq, attn, width, resp)
            entropy_tok = token_entropy_from_logits(logits)
            with reference_mode(policy):
                ref_logp, ref_logits = response_token_logprobs(policy, seq, attn, width, resp)
            # Exact per-position KL(pi_old || pi_ref) over the full vocabulary (>= 0), at the sampled prefixes.
            kl_exact_tok = exact_token_kl(logits, ref_logits)
            del ref_logits
            del logits
            values = token_values(value_model, seq, attn)[:, width - 1: width - 1 + r_len].float()
            raw_reward = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg["reward_max_length"])).float()
        raw_reward = raw_reward.to(mask.device)
        terminated = torch.tensor(gen["terminated_with_eos"], device=mask.device)
        task_reward = raw_reward - eos_penalty * (~terminated).float()

        # ---- 4-5. shaped rewards, GAE --------------------------------------------------
        rewards = shaped_rewards(task_reward, old_logp, ref_logp, mask, beta_kl)
        advantages, returns = compute_gae(rewards, values * mask, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
        adv_white = normalize_advantages(advantages, mask)

        # ---- 6. PPO epochs -------------------------------------------------------------
        ep_stats = []
        for epoch in range(epochs):
            new_logp, _ = response_token_logprobs(policy, seq, attn, width, resp)
            pol_loss, ratio, clip_frac = ppo_policy_loss(new_logp, old_logp, adv_white, mask, eps)
            pol_opt.zero_grad(set_to_none=True)
            pol_loss.backward()
            pg = torch.nn.utils.clip_grad_norm_(pol_params, max_norm)
            pol_ok = grads_are_finite(pol_params) and math.isfinite(float(pg))
            if pol_ok:
                pol_opt.step()
            pol_opt.zero_grad(set_to_none=True)

            new_values = token_values(value_model, seq, attn)[:, width - 1: width - 1 + r_len].float()
            v_loss = value_mse_loss(new_values, returns, mask)
            val_opt.zero_grad(set_to_none=True)
            (value_coef * v_loss).backward()
            vg = torch.nn.utils.clip_grad_norm_(val_params, max_norm)
            val_ok = grads_are_finite(val_params) and math.isfinite(float(vg))
            if val_ok:
                val_opt.step()
            val_opt.zero_grad(set_to_none=True)

            ep_stats.append({
                "policy_loss": float(pol_loss.item()),
                "value_loss": float(v_loss.item()),
                "clip_fraction": float(clip_frac.item()),
                "ratio_abs_dev": float(masked_mean((ratio - 1.0).abs(), mask).item()),
                "ratio_max": float(ratio[mask.bool()].max().item()),
                "ratio_min": float(ratio[mask.bool()].min().item()),
                "grad_norm_policy": finite_or_none(pg),
                "grad_norm_value": finite_or_none(vg),
                "skipped_policy_step": not pol_ok,
                "skipped_value_step": not val_ok,
            })
            del new_logp, new_values, pol_loss, v_loss

        def mean_of(key):
            vals = [s[key] for s in ep_stats if s[key] is not None]
            return float(np.mean(vals)) if vals else None

        kl_tok = float(sampled_kl(old_logp, ref_logp, mask).item())
        rec = {
            "run": run_name,
            "update": u + 1,
            "prompt_ids": [str(r.get("prompt_id")) for r in rows],
            "source_indices": [r.get("source_index") for r in rows],
            "reward_raw": float(raw_reward.mean().item()),
            "reward_effective": float(task_reward.mean().item()),
            "kl_token_mean": kl_tok,
            "kl_seq_sum": float(((old_logp - ref_logp) * mask).sum(-1).mean().item()),
            "kl_exact_token_mean": float(masked_mean(kl_exact_tok, mask).item()),
            "kl_exact_seq_sum": float((kl_exact_tok * mask).sum(-1).mean().item()),
            "policy_loss": mean_of("policy_loss"),
            "value_loss": mean_of("value_loss"),
            "value_loss_first_epoch": ep_stats[0]["value_loss"],
            "entropy": float(masked_mean(entropy_tok, mask).item()),
            "sampled_entropy": float(sample_entropy(old_logp, mask).item()),
            "grad_norm_policy": mean_of("grad_norm_policy"),
            "grad_norm_value": mean_of("grad_norm_value"),
            "clip_fraction": mean_of("clip_fraction"),
            "clip_fraction_last_epoch": ep_stats[-1]["clip_fraction"],
            "ratio_abs_dev_last_epoch": ep_stats[-1]["ratio_abs_dev"],
            "ratio_max": max(s["ratio_max"] for s in ep_stats),
            "ratio_min": min(s["ratio_min"] for s in ep_stats),
            "response_length": float(mask.sum(-1).mean().item()),
            "terminated_with_eos": float(terminated.float().mean().item()),
            "truncated": float(np.mean(gen["truncated"])),
            "value_mean": float(masked_mean(values, mask).item()),
            "return_mean": float(masked_mean(returns, mask).item()),
            "advantage_mean_raw": float(masked_mean(advantages, mask).item()),
            "critic_explained_variance": explained_variance(values, returns, mask),
            "skipped_steps": sum(int(s["skipped_policy_step"]) + int(s["skipped_value_step"]) for s in ep_stats),
            "epochs": ep_stats,
            "elapsed_sec": timer(),
            "peak_vram_gib": peak_vram_gib(),
        }
        append_jsonl(log_path, rec)
        for i, r in enumerate(rows):
            append_jsonl(roll_path, {
                "update": u + 1, "prompt_id": str(r.get("prompt_id")), "source_index": r.get("source_index"),
                "response": gen["responses"][i], "response_tokens": int(gen["response_lengths"][i]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][i]), "truncated": bool(gen["truncated"][i]),
                "raw_reward": float(raw_reward[i].item()), "effective_reward": float(task_reward[i].item()),
            })
        print(f"[PPO:{run_name}] update {u + 1}/{n_updates} reward={rec['reward_raw']:.3f} kl={kl_tok:.4f} kl_exact={rec['kl_exact_token_mean']:.2e} "
              f"pl={rec['policy_loss']:.4f} vl={rec['value_loss']:.4f} H={rec['entropy']:.3f} "
              f"clip={rec['clip_fraction_last_epoch']:.3f} len={rec['response_length']:.0f} t={rec['elapsed_sec']:.0f}s", flush=True)
        del seq, attn, resp, mask, old_logp, ref_logp, kl_exact_tok, values, rewards, advantages, returns, adv_white
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    value_model.save_pretrained(str(out / "critic_adapter"))
    summary = {
        "run_name": run_name,
        "updates": n_updates,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "prompts_per_update": P,
        "ppo_epochs": epochs,
        "used_prompt_ids": used_prompt_ids,
        "dropped_long_prompt_ids": dropped,
        "wall_clock_sec": timer(),
        "peak_vram_gib": peak_vram_gib(),
        "gpu": gpu_name(),
        "output": str(out),
        "start_policy": cfg["paths"]["ppo_midpoint_policy"],
        "start_critic": cfg["paths"]["ppo_midpoint_value"],
        "config": {k: v for k, v in cfg.items() if k != "paths"},
    }
    save_json(out / "run_config.json", summary)
    save_json(results_dir / "train_logs" / f"{run_name}_summary.json", summary)
    print(f"[PPO:{run_name}] done in {summary['wall_clock_sec']:.0f}s, peak VRAM {summary['peak_vram_gib']:.2f} GiB", flush=True)
    del bundle, policy, value_model, rm, pol_opt, val_opt
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
