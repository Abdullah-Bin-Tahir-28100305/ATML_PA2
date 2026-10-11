"""Task 5 Step 2: reward-sensitivity / robustness study on the controlled diagnostic set.

For every problem (20) the set has five variants. Controlled pairs (better response first):
  reasoning (same correct final answer):
    clean_correct  vs corrupt_reasoning_correct_final   -> S_reason
    clean_correct  vs persuasive_filler_correct         -> filler susceptibility
  outcome (designated final answer changes):
    clean_correct  vs good_reasoning_wrong_final        -> S_outcome
    clean_correct  vs gold_distractor_wrong_final       -> S_outcome / distractor robustness
  conflict (reported, no "better" label):
    corrupt_reasoning_correct_final vs good_reasoning_wrong_final

Mechanisms:
  RLVR  - exact verifier reward R(y) = 1[designated final == gold]; a pair is a tie if R is equal.
  RLAIF - the fixed pairwise AI judge; each pair is judged in both A/B orders (the judge's own
          hash-based orientation swap still applies); the primary result uses the (better, worse)
          call, and orientation consistency between the two calls is reported. Also the RLAIF
          group reward (wins + 0.5 ties)/(K-1) over all five variants of a problem, i.e. the reward
          an RLAIF policy would receive for each variant.
For each pair category and mechanism: better-response rate, tie rate, wrong-preference rate.
S_reason = Pr[R(clean) > R(reason-corrupt)], S_outcome = Pr[R(correct final) > R(wrong final)].

Writes results/task5_feedback/perturbations/{pair_scores.jsonl, summary.json, summary.csv}.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

PAIRS = [
    # (category, kind, better, worse)
    ("reasoning_corruption", "reasoning", "clean_correct", "corrupt_reasoning_correct_final"),
    ("persuasive_filler", "reasoning", "clean_correct", "persuasive_filler_correct"),
    ("wrong_final", "outcome", "clean_correct", "good_reasoning_wrong_final"),
    ("gold_distractor", "outcome", "clean_correct", "gold_distractor_wrong_final"),
    ("conflict_correct_final_bad_reasoning_vs_good_reasoning_wrong_final", "conflict",
     "corrupt_reasoning_correct_final", "good_reasoning_wrong_final"),
]
VARIANT_ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "persuasive_filler_correct",
                 "good_reasoning_wrong_final", "gold_distractor_wrong_final"]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def outcome_of(pref):
    return {"better": "better", "worse": "wrong", "tie": "tie"}[pref]


def rates(prefs):
    n = len(prefs)
    return {
        "n": n,
        "better_rate": float(np.mean([p == "better" for p in prefs])) if n else None,
        "tie_rate": float(np.mean([p == "tie" for p in prefs])) if n else None,
        "wrong_preference_rate": float(np.mean([p == "worse" for p in prefs])) if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--skip-judge", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    print("Use exact_reward(...) for RLVR and PairwiseAIJudge(...) for RLAIF.")
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback" / "perturbations"

    judge = None if args.skip_judge else PairwiseAIJudge(cfg, repo_path(cfg["results_dir"]) / "task5_feedback" / "judge_cache.json")

    # Verifier sanity check against the staff-provided expected reward.
    verifier_rows = []
    for pid, v in groups.items():
        for name, row in v.items():
            r = exact_reward(row["response"], row["gold_final"])
            verifier_rows.append({"problem_id": pid, "variant": name, "verifier_reward": r,
                                  "expected_exact_reward": row.get("expected_exact_reward"),
                                  "matches_expected": r == float(row.get("expected_exact_reward", r))})

    pair_rows = []
    for pid, v in groups.items():
        question = v["clean_correct"]["question"]
        for cat, kind, better, worse in PAIRS:
            yb, yw = v[better]["response"], v[worse]["response"]
            rb, rw = exact_reward(yb, v[better]["gold_final"]), exact_reward(yw, v[worse]["gold_final"])
            ver = "better" if rb > rw else ("worse" if rw > rb else "tie")
            row = {"problem_id": pid, "category": cat, "kind": kind, "better_variant": better, "worse_variant": worse,
                   "verifier_better": rb, "verifier_worse": rw, "verifier_pref": ver}
            if judge is not None:
                p1 = judge.compare(question, yb, yw)           # better as candidate A
                p2 = judge.compare(question, yw, yb)           # reversed order
                j1 = {"A": "better", "B": "worse", "TIE": "tie"}[p1]
                j2 = {"A": "worse", "B": "better", "TIE": "tie"}[p2]
                row.update({"judge_pref": j1, "judge_pref_reversed": j2, "judge_orientation_consistent": j1 == j2,
                            "judge_score_both_orders": (({"better": 1, "tie": 0.5, "worse": 0}[j1]) + ({"better": 1, "tie": 0.5, "worse": 0}[j2])) / 2})
            pair_rows.append(row)
    write_jsonl(outdir / "pair_scores.jsonl", pair_rows)

    group_rows = []
    if judge is not None:
        for pid, v in groups.items():
            responses = [v[name]["response"] for name in VARIANT_ORDER]
            rewards = judge.group_rewards(v["clean_correct"]["question"], responses)
            for name, r in zip(VARIANT_ORDER, rewards):
                group_rows.append({"problem_id": pid, "variant": name, "rlaif_group_reward": r,
                                   "verifier_reward": exact_reward(v[name]["response"], v[name]["gold_final"])})
        write_jsonl(outdir / "rlaif_group_rewards.jsonl", group_rows)

    df = pd.DataFrame(pair_rows)
    summary = {"by_category": {}, "verifier_sanity": {
        "n": len(verifier_rows), "matches_expected_rate": float(np.mean([r["matches_expected"] for r in verifier_rows])),
        "mismatches": [r for r in verifier_rows if not r["matches_expected"]]}}
    table = []
    for cat, kind, better, worse in PAIRS:
        d = df[df["category"] == cat]
        entry = {"kind": kind, "better_variant": better, "worse_variant": worse,
                 "rlvr": rates(d["verifier_pref"].tolist())}
        if judge is not None:
            entry["rlaif"] = rates(d["judge_pref"].tolist())
            entry["rlaif_reversed_order"] = rates(d["judge_pref_reversed"].tolist())
            entry["rlaif_orientation_consistency"] = float(d["judge_orientation_consistent"].mean())
            entry["rlaif_mean_score_both_orders"] = float(d["judge_score_both_orders"].mean())
        summary["by_category"][cat] = entry
        for mech in ["rlvr", "rlaif"]:
            if mech in entry:
                table.append({"category": cat, "kind": kind, "mechanism": mech, **entry[mech],
                              "orientation_consistency": entry.get("rlaif_orientation_consistency") if mech == "rlaif" else None})

    def s_metric(kind_cats, col):
        d = df[df["category"].isin(kind_cats)]
        return float((d[col] == "better").mean()) if len(d) else None

    summary["S_reason"] = {"rlvr": s_metric(["reasoning_corruption"], "verifier_pref")}
    summary["S_outcome"] = {"rlvr": s_metric(["wrong_final", "gold_distractor"], "verifier_pref")}
    summary["S_outcome_by_category"] = {"rlvr": {c: s_metric([c], "verifier_pref") for c in ["wrong_final", "gold_distractor"]}}
    if judge is not None:
        summary["S_reason"]["rlaif"] = s_metric(["reasoning_corruption"], "judge_pref")
        summary["S_outcome"]["rlaif"] = s_metric(["wrong_final", "gold_distractor"], "judge_pref")
        summary["S_outcome_by_category"]["rlaif"] = {c: s_metric([c], "judge_pref") for c in ["wrong_final", "gold_distractor"]}
        summary["filler_susceptibility_rlaif"] = float((df[df["category"] == "persuasive_filler"]["judge_pref"] == "worse").mean())
        g = pd.DataFrame(group_rows)
        summary["mean_reward_by_variant"] = g.groupby("variant")[["rlaif_group_reward", "verifier_reward"]].mean().to_dict()
    save_json(outdir / "summary.json", summary)
    pd.DataFrame(table).to_csv(outdir / "summary.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(pd.DataFrame(table))
    print("S_reason:", summary["S_reason"], " S_outcome:", summary["S_outcome"])


if __name__ == "__main__":
    main()
