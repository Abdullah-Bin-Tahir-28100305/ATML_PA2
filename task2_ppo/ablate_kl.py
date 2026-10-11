"""Task 2 Step 3: reward-overoptimisation / KL-pressure study.

For each beta_KL in configs/ppo.yaml `kl_values`, run the matched short fork (same midpoint,
same clip_epsilon, seed, prompts, generation settings and update budget), then evaluate with the
common held-out protocol. Per-update trajectories (reward, KL, entropy, length) are in
results/task2_ppo/train_logs/<fork>.jsonl; this script collects them into one file for plotting.

Resumable. Writes results/task2_ppo/kl_summary.{json,csv} and kl_trajectories.csv.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task2_ppo.forks import ensure_fork, ensure_midpoint_eval, eval_row, training_stability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-eval-prompts", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    ensure_midpoint_eval(args.config, args.batch_size, args.max_eval_prompts)
    rows, traj = [], []
    eps = float(cfg["clip_epsilon"])
    for beta in cfg["kl_values"]:
        name = ensure_fork(args.config, eps, float(beta), args.batch_size, args.max_eval_prompts, args.force)
        row = {"kl_beta": float(beta), "clip_epsilon": eps, "run": name}
        row.update(eval_row(args.config, name))
        row.update({f"train_{k}": v for k, v in training_stability(args.config, name).items()})
        rows.append(row)
        for r in read_jsonl(results / "train_logs" / f"{name}.jsonl"):
            traj.append({"kl_beta": float(beta), "update": r["update"], "reward_raw": r["reward_raw"],
                         "reward_effective": r["reward_effective"], "kl_token_mean": r["kl_token_mean"],
                         "kl_exact_token_mean": r.get("kl_exact_token_mean"),
                         "entropy": r["entropy"], "response_length": r["response_length"],
                         "policy_loss": r["policy_loss"], "value_loss": r["value_loss"]})
    rows.append({"kl_beta": None, "clip_epsilon": None, "run": "midpoint (no continuation)", **eval_row(args.config, "midpoint")})
    df = pd.DataFrame(rows)
    df.to_csv(results / "kl_summary.csv", index=False)
    save_json(results / "kl_summary.json", rows)
    pd.DataFrame(traj).to_csv(results / "kl_trajectories.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df)


if __name__ == "__main__":
    main()
