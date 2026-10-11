"""Task 5: combine in-domain, controlled-diagnostic and transfer results into one comparison.

Reads results/task5_feedback/{gsm,transfer}/metrics.json and perturbations/summary.json and the
supplied training metadata (checkpoints/*_metadata.json) and writes
results/task5_feedback/feedback_comparison.{json,csv}:
  * per policy: GSM8K / SVAMP exact accuracy, format compliance, mean length, AI win rate vs SFT,
    verifier-judge agreement, absolute and relative in-domain -> transfer drops, failure types;
  * diagnostic summary (S_reason, S_outcome, filler susceptibility, orientation consistency);
  * feedback-source facts for the coverage/noise/exploitability/cost discussion: training-time
    fraction of zero-std groups and reward std (supplied metadata), judge calls per K-group
    (K(K-1)/2) vs one verifier call per completion, and measured judge latency.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json

POLICIES = ["sft", "rlvr", "rlaif"]


def maybe(path):
    p = repo_path(path)
    return load_json(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    base = repo_path(cfg["results_dir"]) / "task5_feedback"
    gsm, tr = maybe(base / "gsm" / "metrics.json"), maybe(base / "transfer" / "metrics.json")
    pert = maybe(base / "perturbations" / "summary.json")
    if gsm is None or tr is None:
        raise SystemExit("Run evaluate_math for --dataset gsm and --dataset transfer first.")

    rows = []
    for p in POLICIES:
        g, t = gsm[p], tr[p]
        row = {
            "policy": p,
            "gsm_exact_accuracy": g["exact_accuracy"],
            "svamp_exact_accuracy": t["exact_accuracy"],
            "accuracy_drop_abs": g["exact_accuracy"] - t["exact_accuracy"],
            "accuracy_drop_rel": (g["exact_accuracy"] - t["exact_accuracy"]) / g["exact_accuracy"] if g["exact_accuracy"] else None,
            "gsm_format_compliance": g["format_compliance"],
            "svamp_format_compliance": t["format_compliance"],
            "gsm_mean_tokens": g["response_tokens"]["mean"],
            "svamp_mean_tokens": t["response_tokens"]["mean"],
            "gsm_ai_win_rate_vs_sft": g.get("ai_win_rate_vs_sft"),
            "svamp_ai_win_rate_vs_sft": t.get("ai_win_rate_vs_sft"),
            "gsm_verifier_judge_agreement": g.get("verifier_judge_agreement_vs_sft"),
            "svamp_verifier_judge_agreement": t.get("verifier_judge_agreement_vs_sft"),
            "gsm_failure_types": json.dumps(g["failure_types"]),
            "svamp_failure_types": json.dumps(t["failure_types"]),
        }
        if row["gsm_ai_win_rate_vs_sft"] is not None and row["svamp_ai_win_rate_vs_sft"] is not None:
            row["win_rate_drop_abs"] = row["gsm_ai_win_rate_vs_sft"] - row["svamp_ai_win_rate_vs_sft"]
        rows.append(row)

    meta = {name: maybe(f"checkpoints/{name}_metadata.json") for name in ["rlvr", "rlaif"]}
    latency = []
    for m in (gsm, tr):
        for v in m.get("pairwise_vs_judge", {}).values():
            if v.get("mean_judge_seconds_uncached"):
                latency.append(v["mean_judge_seconds_uncached"])
    K = int((meta.get("rlaif") or {}).get("num_generations", 4))
    feedback_source = {
        "training_metadata": meta,
        "rlvr_reward_calls_per_group": K,
        "rlaif_judge_calls_per_group": K * (K - 1) // 2,
        "rlaif_judge_model": cfg["ai_judge_model"],
        "measured_judge_seconds_per_comparison": (sum(latency) / len(latency)) if latency else None,
        "rlvr_reward_cost": "regex parse + numeric comparison (no model call)",
    }
    out = {"policies": rows, "gsm_pairwise": gsm.get("pairwise_vs_judge"), "svamp_pairwise": tr.get("pairwise_vs_judge"),
           "diagnostics": pert, "feedback_source": feedback_source}
    save_json(base / "feedback_comparison.json", out)
    pd.DataFrame(rows).to_csv(base / "feedback_comparison.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(pd.DataFrame(rows).drop(columns=["gsm_failure_types", "svamp_failure_types"]))
    if pert:
        print("S_reason:", pert.get("S_reason"), "S_outcome:", pert.get("S_outcome"))


if __name__ == "__main__":
    main()
