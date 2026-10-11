"""Task 3 Step 1: GRPO continuation from the supplied midpoint adapter.

Per update (configs/grpo.yaml):
  1. next `prompts_per_update` prompts from the seed-fixed order of the UltraFeedback RL pool;
  2. K = `num_generations` sampled completions per prompt (cap `max_completion_length`);
  3. learned reward from the fixed reward model for every completion;
  4. group-relative advantages A_k = (r_k - mean_group) / (std_group + eps)  (task3_grpo.grpo, corrected);
  5. completions that hit the length cap without EOS are masked out of the loss
     (`mask_truncated_completions`); they still count in their group's reward statistics;
  6. `policy_epochs` clipped-ratio updates with a k3 KL penalty to the frozen reference
     (adapter disabled), using `loss_type` grpo (1/T_k per sequence) or dr_grpo (constant 1/L_max).
Logged per update: reward, KL to reference (released sampled estimator), within-group reward std,
fraction of zero-std (uninformative) groups, policy loss, gradient norm, entropy, response length,
and (optionally) per-completion gradient-norm diagnostics used by the normalisation study.
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
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
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
from task3_grpo.grpo import (
    group_relative_advantages,
    group_reward_stats,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    disable_dropout(policy)
    upcast_trainable_params(policy)
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def per_completion_grad_norms(policy, params, seq, attn, width, resp, old_logp, adv, mask, ref_logp, eps, loss_type, max_len):
    """Gradient norm of each completion's policy-term contribution to the batch loss.

    Contribution of completion i = -(per-sequence objective_i) / N, with per-sequence objective
    normalised by 1/T_i (grpo) or 1/L_max (dr_grpo); the KL term is excluded so the statistic
    isolates the length-normalisation effect. Uses extra forward/backward passes; gradients are
    discarded before the real update.
    """
    N = seq.shape[0]
    norms = []
    for i in range(N):
        if float(mask[i].sum()) == 0:
            norms.append(0.0)
            continue
        policy.zero_grad(set_to_none=True)
        new_i, _ = response_token_logprobs(policy, seq[i:i + 1], attn[i:i + 1], width, resp[i:i + 1])
        loss_i, _ = grpo_policy_loss(new_i, old_logp[i:i + 1], adv[i:i + 1], mask[i:i + 1], ref_logp[i:i + 1],
                                     eps, 0.0, loss_type=loss_type, max_completion_length=max_len)
        (loss_i / N).backward()
        sq = sum(float(p.grad.float().pow(2).sum()) for p in params if p.grad is not None)
        norms.append(math.sqrt(sq))
    policy.zero_grad(set_to_none=True)
    return norms


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", grad_diagnostics: bool = False):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.mkdir(parents=True, exist_ok=True)

    tokenizer, policy, opt = bundle["tokenizer"], bundle["policy"], bundle["optimizer"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    params = trainable_parameters(policy)
    seed = int(cfg["seed"])
    n_updates = int(cfg["updates"])
    P = int(cfg["prompts_per_update"])
    K = int(cfg["num_generations"])
    eps = float(cfg["clip_epsilon"])
    beta = float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    epochs = int(cfg.get("policy_epochs", 1))
    max_norm = float(cfg.get("max_grad_norm", 1.0))
    gen_cfg = cfg["generation"]

    schedule, dropped = rl_prompt_schedule(bundle["prompt_rows"], tokenizer, int(cfg["max_prompt_length"]), seed)
    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / "train_logs" / f"{run_name}.jsonl"
    roll_path = results_dir / "rollouts" / f"{run_name}.jsonl"
    for p in (log_path, roll_path):
        if p.exists():
            p.unlink()
    print(f"[GRPO:{run_name}] loss_type={loss_type} updates={n_updates} prompts/update={P} K={K} eps={eps} "
          f"beta={beta} usable_prompts={len(schedule)} (dropped {len(dropped)})", flush=True)

    timer = Timer()
    reset_peak_vram()
    policy.train()
    used_prompt_ids = []
    for u in range(n_updates):
        rows = [schedule[(u * P + j) % len(schedule)] for j in range(P)]
        used_prompt_ids += [str(r.get("prompt_id")) for r in rows]
        msgs = [prompt_messages(r) for r in rows for _ in range(K)]
        group_ids = torch.tensor([j for j in range(P) for _ in range(K)])

        torch.manual_seed(seed * 1000 + u)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed * 1000 + u)
        gen = batch_generate(
            policy, tokenizer, msgs,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=max_len,
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
        full_mask = gen["response_mask"][:, :r_len].float().clone()
        loss_mask = mask_truncated_sequences(full_mask, gen["truncated"]) if cfg.get("mask_truncated_completions", True) else full_mask

        with torch.no_grad():
            rewards = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg.get("reward_max_length", 1280))).float().to(full_mask.device)
            with reference_mode(policy):
                ref_logp, ref_logits = response_token_logprobs(policy, seq, attn, width, resp)
                del ref_logits
            if epochs > 1:
                old_logp, _ = response_token_logprobs(policy, seq, attn, width, resp)
            else:
                old_logp = None
        adv = group_relative_advantages(rewards, group_ids.to(rewards.device))
        gstats = group_reward_stats(rewards, group_ids.to(rewards.device))

        grad_norms_i = None
        if grad_diagnostics:
            with torch.no_grad():
                old_d = old_logp if old_logp is not None else response_token_logprobs(policy, seq, attn, width, resp)[0]
            grad_norms_i = per_completion_grad_norms(policy, params, seq, attn, width, resp, old_d, adv, loss_mask,
                                                     ref_logp, eps, loss_type, max_len)

        ep_stats = []
        for epoch in range(epochs):
            new_logp, logits = response_token_logprobs(policy, seq, attn, width, resp)
            if epoch == 0:
                with torch.no_grad():
                    entropy_tok = token_entropy_from_logits(logits.detach())
            del logits
            old = new_logp.detach() if old_logp is None else old_logp
            loss, diag = grpo_policy_loss(new_logp, old, adv, loss_mask, ref_logp, eps, beta,
                                          loss_type=loss_type, max_completion_length=max_len)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
            ok = grads_are_finite(params) and math.isfinite(float(gn)) and math.isfinite(float(loss))
            if ok:
                opt.step()
            opt.zero_grad(set_to_none=True)
            ep_stats.append({"loss": float(loss.item()), "grad_norm": finite_or_none(gn), "skipped": not ok,
                             **{k: float(v.item()) for k, v in diag.items()}})
            if epoch == 0:
                kl_ref_tok = float(sampled_kl(new_logp.detach(), ref_logp, full_mask).item())
                samp_ent = float(sample_entropy(new_logp.detach(), full_mask).item())
            del new_logp, loss

        lengths = full_mask.sum(-1)
        rec = {
            "run": run_name,
            "loss_type": loss_type,
            "update": u + 1,
            "prompt_ids": [str(r.get("prompt_id")) for r in rows],
            "source_indices": [r.get("source_index") for r in rows],
            "reward_mean": float(rewards.mean().item()),
            "reward_std_all": float(rewards.std(unbiased=False).item()),
            "group_reward_std_mean": gstats["group_reward_std_mean"],
            "frac_zero_std_groups": gstats["frac_zero_std_groups"],
            "kl_token_mean": kl_ref_tok,
            "kl_k3_penalty": float(np.mean([s["sampled_kl"] for s in ep_stats])),
            "policy_loss": float(np.mean([s["loss"] for s in ep_stats])),
            "policy_term": float(np.mean([s["policy_term"] for s in ep_stats])),
            "grad_norm": finite_or_none(np.mean([s["grad_norm"] for s in ep_stats if s["grad_norm"] is not None]) if any(s["grad_norm"] is not None for s in ep_stats) else None),
            "entropy": float(masked_mean(entropy_tok, full_mask).item()),
            "sampled_entropy": samp_ent,
            "clip_fraction": float(np.mean([s["clip_fraction"] for s in ep_stats])),
            "response_length": float(lengths.mean().item()),
            "truncated_fraction": float(np.mean(gen["truncated"])),
            "masked_completions": int(sum(gen["truncated"])) if cfg.get("mask_truncated_completions", True) else 0,
            "skipped_steps": int(sum(s["skipped"] for s in ep_stats)),
            "completions": [
                {"length": int(lengths[i].item()), "reward": float(rewards[i].item()), "advantage": float(adv[i].item()),
                 "truncated": bool(gen["truncated"][i]), "masked": bool(float(loss_mask[i].sum()) == 0),
                 "grad_norm": None if grad_norms_i is None else grad_norms_i[i]}
                for i in range(len(msgs))
            ],
            "elapsed_sec": timer(),
            "peak_vram_gib": peak_vram_gib(),
        }
        append_jsonl(log_path, rec)
        for i in range(len(msgs)):
            r = rows[i // K]
            append_jsonl(roll_path, {"update": u + 1, "prompt_id": str(r.get("prompt_id")), "source_index": r.get("source_index"),
                                     "generation_index": i % K, "response": gen["responses"][i],
                                     "response_tokens": int(gen["response_lengths"][i]), "truncated": bool(gen["truncated"][i]),
                                     "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                                     "reward": float(rewards[i].item()), "advantage": float(adv[i].item())})
        print(f"[GRPO:{run_name}] update {u + 1}/{n_updates} reward={rec['reward_mean']:.3f} grp_std={rec['group_reward_std_mean']:.3f} "
              f"kl={rec['kl_token_mean']:.4f} loss={rec['policy_loss']:.4f} gn={rec['grad_norm']} H={rec['entropy']:.3f} "
              f"len={rec['response_length']:.0f} trunc={rec['truncated_fraction']:.2f} t={rec['elapsed_sec']:.0f}s", flush=True)
        del seq, attn, resp, full_mask, loss_mask, ref_logp, old_logp, rewards, adv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "updates": n_updates,
        "used_prompt_ids": used_prompt_ids,
        "dropped_long_prompt_ids": dropped,
        "wall_clock_sec": timer(),
        "peak_vram_gib": peak_vram_gib(),
        "gpu": gpu_name(),
        "output": str(out),
        "start_policy": cfg["paths"]["grpo_midpoint_policy"],
        "grad_diagnostics": grad_diagnostics,
        "config": {k: v for k, v in cfg.items() if k != "paths"},
    }
    save_json(out / "run_config.json", summary)
    save_json(results_dir / "train_logs" / f"{run_name}_summary.json", summary)
    print(f"[GRPO:{run_name}] done in {summary['wall_clock_sec']:.0f}s, peak VRAM {summary['peak_vram_gib']:.2f} GiB", flush=True)
    del bundle, policy, opt, rm
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--grad-diagnostics", action="store_true", help="also log per-completion gradient norms")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.grad_diagnostics)


if __name__ == "__main__":
    main()
