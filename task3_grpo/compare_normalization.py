"""Task 3 Step 3: canonical GRPO (1/T_k sequence normalisation) vs Dr. GRPO-style (constant 1/L_max).

Both forks start from the identical supplied midpoint and use the same seed, prompts, K,
reward model, beta, epsilon, generation settings and `fork_updates` budget; only `loss_type`
changes. Each fork logs per-completion (length, advantage, gradient norm of its contribution),
which gives the length-conditioned statistics:
  * Spearman/Pearson correlation between completion length and per-completion gradient norm;
  * mean gradient norm for short vs long completions (split at the median length of each fork);
  * implied per-token weight |A|/T (grpo) vs |A|/L_max (dr_grpo), and the share of total
    |A|-weighted token mass falling on above-median-length completions.
Then both forks are evaluated with the common held-out protocol (reward, KL, length).

Resumable. Writes results/task3_grpo/normalization_summary.{json,csv}.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from common.training_utils import adapter_exists
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_grpo_policy


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def length_conditioned(log, loss_type, max_len):
    comps = [c for r in log for c in r["completions"] if not c["masked"]]
    if not comps:
        return {}
    L = np.array([c["length"] for c in comps], float)
    A = np.abs(np.array([c["advantage"] for c in comps], float))
    G = np.array([c["grad_norm"] if c["grad_norm"] is not None else np.nan for c in comps], float)
    med = float(np.median(L))
    short, long_ = L <= med, L > med
    per_token_w = A / L if loss_type == "grpo" else A / float(max_len)
    token_mass = per_token_w * L  # total weight a completion's tokens receive in the loss
    out = {
        "n_completions": int(len(L)),
        "median_length": med,
        "corr_length_gradnorm_spearman": spearman(L[~np.isnan(G)], G[~np.isnan(G)]),
        "corr_length_gradnorm_pearson": pearson(L[~np.isnan(G)], G[~np.isnan(G)]),
        "gradnorm_short_mean": float(np.nanmean(G[short])) if short.any() else None,
        "gradnorm_long_mean": float(np.nanmean(G[long_])) if long_.any() else None,
        "per_token_weight_short_mean": float(per_token_w[short].mean()) if short.any() else None,
        "per_token_weight_long_mean": float(per_token_w[long_].mean()) if long_.any() else None,
        "share_of_weight_on_long": float(token_mass[long_].sum() / max(token_mass.sum(), 1e-12)),
        "corr_length_per_token_weight": spearman(L, per_token_w),
    }
    if out["gradnorm_short_mean"] and out["gradnorm_long_mean"]:
        out["gradnorm_long_over_short"] = out["gradnorm_long_mean"] / out["gradnorm_short_mean"]
    return out


def train_trajectory_stats(log):
    get = lambda k: np.array([r[k] for r in log if r.get(k) is not None], float)
    return {
        "train_reward_mean": float(get("reward_mean").mean()),
        "train_kl_last": float(get("kl_token_mean")[-1]),
        "train_length_first": float(get("response_length")[0]),
        "train_length_last": float(get("response_length")[-1]),
        "train_length_mean": float(get("response_length").mean()),
        "train_grad_norm_mean": float(get("grad_norm").mean()) if get("grad_norm").size else None,
        "train_entropy_last": float(get("entropy")[-1]),
        "train_frac_zero_std_groups": float(get("frac_zero_std_groups").mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-eval-prompts", type=int)
    ap.add_argument("--no-grad-diagnostics", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    if not (results / "eval_midpoint.json").exists():
        evaluate_grpo_policy(args.config, cfg["paths"]["grpo_midpoint_policy"], "midpoint", args.batch_size, args.max_eval_prompts)

    rows = []
    for loss_type in ["grpo", "dr_grpo"]:
        name = f"fork_{loss_type}"
        out = f"outputs/task3_grpo/forks/{name}"
        if args.force or not adapter_exists(out):
            run_grpo(args.config, output=out, updates=int(cfg["fork_updates"]), loss_type=loss_type,
                     run_name=name, grad_diagnostics=not args.no_grad_diagnostics)
        if args.force or not (results / f"eval_{name}.json").exists():
            evaluate_grpo_policy(args.config, out, name, args.batch_size, args.max_eval_prompts)
        log = read_jsonl(results / "train_logs" / f"{name}.jsonl")
        ev = load_json(results / f"eval_{name}.json")
        rows.append({
            "loss_type": loss_type,
            "run": name,
            "heldout_reward_mean": ev["reward_mean"],
            "heldout_kl_token_mean": ev["kl_token_mean"],
            "heldout_kl_seq_mean": ev["kl_seq_mean"],
            "heldout_len_mean": ev["length"]["mean"],
            "heldout_len_std": ev["length"]["std"],
            "heldout_len_iqr": ev["length"]["iqr"],
            "heldout_entropy": ev["entropy_token_mean"],
            "heldout_truncation_rate": ev["truncation_rate"],
            **train_trajectory_stats(log),
            **{f"lc_{k}": v for k, v in length_conditioned(log, loss_type, int(cfg["max_completion_length"])).items()},
        })
    mid = load_json(results / "eval_midpoint.json")
    rows.append({"loss_type": None, "run": "midpoint (no continuation)", "heldout_reward_mean": mid["reward_mean"],
                 "heldout_kl_token_mean": mid["kl_token_mean"], "heldout_kl_seq_mean": mid["kl_seq_mean"],
                 "heldout_len_mean": mid["length"]["mean"], "heldout_len_std": mid["length"]["std"],
                 "heldout_len_iqr": mid["length"]["iqr"], "heldout_entropy": mid["entropy_token_mean"],
                 "heldout_truncation_rate": mid["truncation_rate"]})
    df = pd.DataFrame(rows)
    df.to_csv(results / "normalization_summary.csv", index=False)
    save_json(results / "normalization_summary.json", rows)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.T)


if __name__ == "__main__":
    main()
