"""Task 2 Step 2: clipping study.

Part A - cached rollout batch (cached/ppo_rollout.pt, 32 rollouts from the midpoint):
  * rebuild prompt+response token ids (prompt from the fixed RL prompt pools via prompt_id;
    response re-tokenised and checked against the cached log-prob length);
  * KL-shaped rewards from the cached old/reference log-probs (beta_KL from the config),
    GAE with the cached critic values, whitened advantages;
  * (A1, static) ratio of the supplied midpoint policy to the cached old log-probs, and the
    clip fraction / affected-token fraction / clipped vs unclipped surrogate for each epsilon;
  * (A2, immediate update geometry) for each epsilon, start from the identical midpoint and run
    `ppo_epochs` passes of minibatch PPO updates on the cached batch with that epsilon, recording
    clip fraction, affected-token fraction and surrogate values before every step and after the
    last step (same response mask throughout).
Part B - matched short forks: for each epsilon, `fork_updates` online updates from the same
midpoint with the same reward/KL settings, then the common held-out evaluation.

Resumable. Writes results/task2_ppo/clipping_cached.json and clipping_summary.{json,csv}.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from common.training_utils import device_of, disable_dropout, upcast_trainable_params
from task2_ppo.forks import ensure_fork, ensure_midpoint_eval, eval_row, training_stability
from task2_ppo.ppo import clip_diagnostics, compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def rebuild_batch(cfg, rows, tokenizer):
    """Token ids + aligned cached tensors for every rollout whose re-tokenisation matches."""
    pool = {}
    for key in ("rl_prompt_eval", "rl_prompt_train"):
        for r in read_jsonl(cfg["paths"][key]):
            pool.setdefault(str(r["prompt_id"]), r)
    items, skipped = [], []
    for row in rows:
        pid = str(row.get("prompt_id"))
        if pid not in pool:
            skipped.append({"prompt_id": pid, "reason": "prompt_not_found"})
            continue
        prompt_ids = tokenizer.apply_chat_template(prompt_messages(pool[pid]), tokenize=True, add_generation_prompt=True)
        body = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        n = len(row["old_logprobs"])
        candidates = [body + [tokenizer.eos_token_id], body] if row.get("terminated_with_eos", True) else [body, body + [tokenizer.eos_token_id]]
        resp = next((c for c in candidates if len(c) == n), None)
        if resp is None:
            skipped.append({"prompt_id": pid, "reason": f"length_mismatch cached={n} retokenized={len(body)}"})
            continue
        items.append({
            "prompt_id": pid,
            "source_index": row.get("source_index"),
            "prompt_ids": prompt_ids,
            "response_ids": resp,
            "old": torch.as_tensor(row["old_logprobs"], dtype=torch.float32),
            "ref": torch.as_tensor(row["ref_logprobs"], dtype=torch.float32),
            "values": torch.as_tensor(row["values"], dtype=torch.float32) if "values" in row else None,
            "reward": float(row.get("effective_terminal_reward", row.get("raw_terminal_reward", 0.0))),
        })
    return items, skipped


def collate_items(items, tokenizer, device):
    """Left-pad prompts to a common width and right-pad responses, matching batch_generate layout."""
    width = max(len(it["prompt_ids"]) for it in items)
    R = max(len(it["response_ids"]) for it in items)
    pad = tokenizer.pad_token_id
    B = len(items)
    seq = torch.full((B, width + R), pad, dtype=torch.long)
    attn = torch.zeros((B, width + R), dtype=torch.long)
    resp = torch.full((B, R), pad, dtype=torch.long)
    mask = torch.zeros((B, R))
    old = torch.zeros((B, R))
    ref = torch.zeros((B, R))
    vals = torch.zeros((B, R))
    rew = torch.zeros(B)
    for i, it in enumerate(items):
        p, r = it["prompt_ids"], it["response_ids"]
        seq[i, width - len(p): width] = torch.tensor(p)
        attn[i, width - len(p): width + len(r)] = 1
        seq[i, width: width + len(r)] = torch.tensor(r)
        resp[i, : len(r)] = torch.tensor(r)
        mask[i, : len(r)] = 1
        old[i, : len(r)] = it["old"]
        ref[i, : len(r)] = it["ref"]
        if it["values"] is not None:
            vals[i, : len(r)] = it["values"]
        rew[i] = it["reward"]
    t = lambda x: x.to(device)
    return {"seq": t(seq), "attn": t(attn), "width": width, "resp": t(resp), "mask": t(mask),
            "old": t(old), "ref": t(ref), "values": t(vals), "reward": t(rew)}


def logprobs_in_chunks(policy, batch, idx, chunk=2, grad=False):
    outs = []
    for s in range(0, len(idx), chunk):
        sl = idx[s:s + chunk]
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            lp, _ = response_token_logprobs(policy, batch["seq"][sl], batch["attn"][sl], batch["width"], batch["resp"][sl])
        outs.append(lp)
    return torch.cat(outs, 0)


def cached_study(cfg, args):
    tokenizer = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))
    items, skipped = rebuild_batch(cfg, rows, tokenizer)
    print(f"Rebuilt {len(items)} rollouts; skipped {len(skipped)}")
    if not items:
        raise SystemExit("No cached rollouts could be aligned; see skipped reasons: " + str(skipped[:5]))

    set_seed(int(cfg["seed"]))
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    disable_dropout(policy)
    upcast_trainable_params(policy)
    device = device_of(policy)
    batch = collate_items(items, tokenizer, device)
    mask = batch["mask"]

    rewards = shaped_rewards(batch["reward"], batch["old"], batch["ref"], mask, float(cfg["kl_beta"]))
    adv, returns = compute_gae(rewards, batch["values"] * mask, mask, float(cfg["gamma"]), float(cfg["gae_lambda"]))
    adv = normalize_advantages(adv, mask)
    B = len(items)
    all_idx = list(range(B))

    # A1: static geometry of the supplied midpoint relative to the cached behaviour policy.
    new0 = logprobs_in_chunks(policy, batch, all_idx)
    ratio0 = torch.exp(new0 - batch["old"])
    lr0 = (new0 - batch["old"])[mask.bool()]
    static = {
        "ratio_stats": {
            "mean_abs_log_ratio": float(lr0.abs().mean()),
            "max_abs_log_ratio": float(lr0.abs().max()),
            "ratio_q01": float(torch.quantile(ratio0[mask.bool()], 0.01)),
            "ratio_q99": float(torch.quantile(ratio0[mask.bool()], 0.99)),
        },
        "per_epsilon": {f"{e:g}": clip_diagnostics(ratio0, adv, mask, float(e)) for e in cfg["clip_values"]},
    }

    # A2: minibatch PPO updates on the fixed cached batch, from the same start, per epsilon.
    init_state = {k: v.detach().clone() for k, v in policy.state_dict().items() if "lora_" in k}
    params = trainable_parameters(policy)
    mb = int(args.minibatch_size)
    rng = np.random.default_rng(int(cfg["seed"]))
    perms = [rng.permutation(B).tolist() for _ in range(int(cfg["ppo_epochs"]))]
    dynamic = {}
    for e in cfg["clip_values"]:
        e = float(e)
        policy.load_state_dict(init_state, strict=False)
        opt = AdamW(params, lr=float(cfg["policy_learning_rate"]) * float(args.lr_scale))
        steps = []
        for ep, perm in enumerate(perms):
            for s in range(0, B, mb):
                idx = perm[s:s + mb]
                m_tokens = float(mask[idx].sum())
                opt.zero_grad(set_to_none=True)
                diag_parts = []
                for c in range(0, len(idx), 2):
                    sl = idx[c:c + 2]
                    new = logprobs_in_chunks(policy, batch, sl, chunk=2, grad=True)
                    loss, ratio, _ = ppo_policy_loss(new, batch["old"][sl], adv[sl], mask[sl], e)
                    # masked_mean over the micro-batch -> rescale to a token mean over the minibatch.
                    (loss * float(mask[sl].sum()) / m_tokens).backward()
                    diag_parts.append((ratio, sl))
                ratio_mb = torch.cat([r for r, _ in diag_parts], 0)
                sl_all = [i for _, sl in diag_parts for i in sl]
                d = clip_diagnostics(ratio_mb, adv[sl_all], mask[sl_all], e)
                gn = torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
                opt.step()
                steps.append({"epoch": ep, "minibatch_start": s, "grad_norm": float(gn), **d})
        newT = logprobs_in_chunks(policy, batch, all_idx)
        ratioT = torch.exp(newT - batch["old"])
        final = clip_diagnostics(ratioT, adv, mask, e)
        final["mean_abs_log_ratio"] = float((newT - batch["old"])[mask.bool()].abs().mean())
        final["kl_old_to_new_token_mean"] = float(masked_mean(batch["old"] - newT, mask))
        dynamic[f"{e:g}"] = {
            "steps": steps,
            "final_full_batch": final,
            "mean_clip_fraction_over_steps": float(np.mean([x["clip_fraction"] for x in steps])),
            "mean_affected_fraction_over_steps": float(np.mean([x["affected_fraction"] for x in steps])),
        }
        print(f"[clip-cached] eps={e:g} final clip={final['clip_fraction']:.4f} affected={final['affected_fraction']:.4f} "
              f"|log rho|={final['mean_abs_log_ratio']:.5f}")
        del opt
    del policy
    clear_gpu()
    return {
        "n_cached": len(rows),
        "n_aligned": B,
        "skipped": skipped,
        "kl_beta": float(cfg["kl_beta"]),
        "minibatch_size": mb,
        "ppo_epochs": int(cfg["ppo_epochs"]),
        "learning_rate": float(cfg["policy_learning_rate"]) * float(args.lr_scale),
        "advantage_whitening": True,
        "static_midpoint_vs_cache": static,
        "after_updates": dynamic,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--minibatch-size", type=int, default=8)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiplier on policy_learning_rate for the cached-batch updates")
    ap.add_argument("--skip-cached", action="store_true")
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16, help="generation batch size for evaluation")
    ap.add_argument("--max-eval-prompts", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])

    cached_path = results / "clipping_cached.json"
    if not args.skip_cached and (args.force or not cached_path.exists()):
        save_json(cached_path, cached_study(cfg, args))

    if args.skip_forks:
        return
    ensure_midpoint_eval(args.config, args.batch_size, args.max_eval_prompts)
    rows = []
    cached = None
    if cached_path.exists():
        from common.logging_utils import load_json
        cached = load_json(cached_path)
    for e in cfg["clip_values"]:
        name = ensure_fork(args.config, float(e), float(cfg["kl_beta"]), args.batch_size, args.max_eval_prompts, args.force)
        row = {"clip_epsilon": float(e), "kl_beta": float(cfg["kl_beta"]), "run": name}
        if cached:
            st = cached["static_midpoint_vs_cache"]["per_epsilon"][f"{float(e):g}"]
            dy = cached["after_updates"][f"{float(e):g}"]
            row.update({
                "cached_static_clip_fraction": st["clip_fraction"],
                "cached_static_affected_fraction": st["affected_fraction"],
                "cached_updated_clip_fraction": dy["final_full_batch"]["clip_fraction"],
                "cached_updated_affected_fraction": dy["final_full_batch"]["affected_fraction"],
                "cached_mean_affected_fraction_over_steps": dy["mean_affected_fraction_over_steps"],
                "cached_clipped_surrogate": dy["final_full_batch"]["clipped_surrogate"],
                "cached_unclipped_surrogate": dy["final_full_batch"]["unclipped_surrogate"],
            })
        row.update(eval_row(args.config, name))
        row.update({f"train_{k}": v for k, v in training_stability(args.config, name).items()})
        rows.append(row)
    mid = eval_row(args.config, "midpoint")
    rows.append({"clip_epsilon": None, "kl_beta": None, "run": "midpoint (no continuation)", **mid})
    df = pd.DataFrame(rows)
    df.to_csv(results / "clipping_summary.csv", index=False)
    save_json(results / "clipping_summary.json", rows)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df)


if __name__ == "__main__":
    main()
