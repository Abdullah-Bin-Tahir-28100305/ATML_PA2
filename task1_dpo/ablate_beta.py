"""Task 1 Step 2: regularisation-strength study.

For each beta in configs/dpo.yaml `betas`, train the short-run DPO configuration from the
original policy initialisation (fresh zero-init LoRA on Qwen2.5-1.5B-Instruct), using the same
first `short_ablation_examples` pairs of the standard training file, the same optimiser, seed,
data order and LoRA configuration. Only beta changes. Each fork is then evaluated with the
identical held-out protocol (same pairs, prompts, decoding, seed).

The script is resumable: finished adapters / evaluations are reused.
Writes results/task1_dpo/beta_ablation_summary.{json,csv}.
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from common.training_utils import adapter_exists
from task1_dpo.evaluate import evaluate_dpo_policy
from task1_dpo.train import run_training


def beta_tag(beta: float) -> str:
    return f"beta_{beta:g}"


def summary_row(label, budget, ev, train_summary=None):
    p = ev["pairs"]["overall"]
    g = ev.get("generation", {})
    length = g.get("length", {})
    return {
        "condition": label,
        "budget": budget,
        "beta": ev["beta"],
        "heldout_dpo_loss": p.get("dpo_loss"),
        "heldout_pref_accuracy": p.get("preference_accuracy"),
        "heldout_margin_mean": p.get("margin_mean"),
        "kl_token_mean": g.get("kl_token_mean"),
        "kl_seq_mean": g.get("kl_seq_mean"),
        "kl_seq_se": g.get("kl_seq_se"),
        "kl_exact_token_mean": g.get("kl_exact_token_mean"),
        "kl_exact_seq_mean": g.get("kl_exact_seq_mean"),
        "reward_mean": g.get("reward_mean"),
        "reward_std": g.get("reward_std"),
        "gen_len_mean": length.get("mean"),
        "gen_len_std": length.get("std"),
        "gen_len_median": length.get("median"),
        "gen_len_iqr": length.get("iqr"),
        "truncation_rate": g.get("truncation_rate"),
        "entropy_token_mean": g.get("entropy_token_mean"),
        "optimizer_steps": (train_summary or {}).get("optimizer_steps"),
        "train_pairs": (train_summary or {}).get("n_pairs"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="*", help="override the beta list")
    ap.add_argument("--batch-size", type=int, default=16, help="generation batch size for evaluation")
    ap.add_argument("--force", action="store_true", help="retrain/re-evaluate even if outputs exist")
    ap.add_argument("--learning-rate", type=float,
                    help="SUPPLEMENTARY sweep at a non-released learning rate; outputs get a _lr<value> suffix "
                         "and a separate summary file, so the required runs are untouched")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = args.betas or [float(b) for b in cfg["betas"]]
    n_short = int(cfg["short_ablation_examples"])
    results = repo_path(cfg["results_dir"])
    print("Required beta values:", betas)
    print("Short-run examples per condition:", n_short)

    lr = args.learning_rate
    suffix = f"_lr{lr:g}" if lr is not None else ""
    lr_label = f", lr={lr:g} (supplementary)" if lr is not None else f", lr={float(cfg['learning_rate']):g} (released)"
    rows = []
    for beta in betas:
        tag = beta_tag(beta) + suffix
        out = f"outputs/task1_dpo/{tag}"
        if args.force or not adapter_exists(out):
            run_training(args.config, tag, output_path=out, beta=beta, max_examples=n_short, learning_rate=lr)
        else:
            print(f"[beta-ablation] reusing trained adapter {out}")
        ev_path = results / f"eval_{tag}.json"
        if args.force or not ev_path.exists():
            evaluate_dpo_policy(args.config, out, tag, beta=beta, batch_size=args.batch_size)
        ev = load_json(ev_path)
        ts = load_json(repo_path(out) / "run_config.json")
        rows.append(summary_row(tag, f"short: first {n_short} pairs, 1 epoch{lr_label}", ev, ts))

    # Reference rows (different budgets, clearly labelled) if they were evaluated.
    for name, budget in [("standard", "standard: full 1500-pair file, 1 epoch"), ("length_balanced", "length-balanced file, 1 epoch"), ("sft", "no DPO (starting policy)")]:
        p = results / f"eval_{name}.json"
        if p.exists():
            ts_path = repo_path(cfg["standard_output"]) / "run_config.json"
            if name == "length_balanced":
                ts_path = repo_path(cfg["length_output"]) / "run_config.json"
            ts = load_json(ts_path) if (name != "sft" and ts_path.exists()) else None
            rows.append(summary_row(name, budget, load_json(p), ts))

    df = pd.DataFrame(rows)
    df.to_csv(results / f"beta_ablation_summary{suffix}.csv", index=False)
    save_json(results / f"beta_ablation_summary{suffix}.json", rows)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)


if __name__ == "__main__":
    main()
