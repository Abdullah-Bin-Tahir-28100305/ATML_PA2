"""Task 3 Step 2: equal-generation group-size study (no training).

The supplied cache holds 8 completions (with rewards) for each of P prompts, i.e. a fixed
budget of 8P generations. For K in {2, 4, 8} the SAME 8P generations are regrouped:

  Partitioning rule (primary): each prompt's 8 completions, ordered by generation_index, are split
  into 8/K consecutive disjoint groups of size K -> 8P/K groups per condition, every cached
  completion used exactly once. Total generations are therefore identical across K; only the
  number of within-prompt comparisons per group changes.

  Secondary estimate: probabilities over ALL C(8, K) subsets of each prompt's completions (lower
  variance estimate of P(informative | prompt, K)), reported alongside.

Metrics per K (overall and per prompt-difficulty bin):
  * informative-group rate: fraction of groups with reward std > tol (tol = task3_grpo.grpo.ZERO_STD_TOL,
    the helper's tolerance); also reported at a few larger practical thresholds;
  * mean within-group reward std;
  * variance of the group-relative signal: variance of (r - group mean) over completions
    (un-normalised relative signal) and variance of the normalised advantages A;
  * fraction of completions with |A| > 0.5 (non-negligible normalised signal).
Difficulty bins (defined once): prompts are ranked by their mean reward over all 8 cached
completions and split into equal-count tertiles: "hard" (lowest mean reward), "medium", "easy".
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import ZERO_STD_TOL, group_relative_advantages

PRACTICAL_THRESHOLDS = [0.05, 0.1, 0.25]


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Each prompt's first 8 completions (sorted by generation_index) are split into 8//k consecutive,
    disjoint groups. Returns a list of dicts {prompt, group_index, rows}.
    """
    if 8 % k != 0:
        raise ValueError("K must divide 8")
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:8]
        for g in range(8 // k):
            groups.append({"prompt": pid, "group_index": g, "rows": rows[g * k:(g + 1) * k]})
    return groups


def difficulty_bins(by_prompt):
    means = {pid: float(np.mean([r["reward"] for r in rows[:8]])) for pid, rows in by_prompt.items()}
    order = sorted(means, key=lambda p: means[p])
    n = len(order)
    labels = {}
    for i, pid in enumerate(order):
        labels[pid] = "hard" if i < n / 3 else ("medium" if i < 2 * n / 3 else "easy")
    return labels, means


def group_metrics(groups):
    stds, rel, advs = [], [], []
    for g in groups:
        r = torch.tensor([float(x["reward"]) for x in g["rows"]])
        stds.append(float(r.std(unbiased=False)))
        rel += (r - r.mean()).tolist()
        advs += group_relative_advantages(r, torch.zeros(len(r), dtype=torch.long)).tolist()
    stds, rel, advs = np.array(stds), np.array(rel), np.array(advs)
    out = {
        "n_groups": int(len(stds)),
        "n_generations": int(len(rel)),
        "informative_rate": float((stds > ZERO_STD_TOL).mean()),
        "mean_group_reward_std": float(stds.mean()),
        "relative_signal_variance": float(rel.var()),
        "normalized_advantage_variance": float(advs.var()),
        "frac_abs_adv_gt_0.5": float((np.abs(advs) > 0.5).mean()),
    }
    for t in PRACTICAL_THRESHOLDS:
        out[f"informative_rate_std_gt_{t:g}"] = float((stds > t).mean())
    return out


def all_subsets_estimate(by_prompt, k):
    """P(group std > tol) and E[std] averaged over all C(8,k) subsets of each prompt."""
    per_prompt = {}
    for pid, rows in by_prompt.items():
        r = np.array([float(x["reward"]) for x in rows[:8]])
        stds = [float(r[list(c)].std()) for c in itertools.combinations(range(8), k)]
        per_prompt[pid] = {"p_informative": float(np.mean(np.array(stds) > ZERO_STD_TOL)), "mean_std": float(np.mean(stds)),
                           **{f"p_std_gt_{t:g}": float(np.mean(np.array(stds) > t)) for t in PRACTICAL_THRESHOLDS}}
    keys = next(iter(per_prompt.values())).keys()
    return {key: float(np.mean([v[key] for v in per_prompt.values()])) for key in keys}, per_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    bins, prompt_means = difficulty_bins(by_prompt)
    all_rows = [r for rows in by_prompt.values() for r in rows[:8]]
    out = {
        "partition_rule": "per prompt, consecutive disjoint groups of size K over generation_index 0..7",
        "difficulty_rule": "tertiles of per-prompt mean reward over all 8 cached completions (hard = lowest)",
        "zero_std_tolerance": ZERO_STD_TOL,
        "total_generations": len(all_rows),
        "n_prompts": len(by_prompt),
        "cache_stats": {
            "reward_mean": float(np.mean([r["reward"] for r in all_rows])),
            "reward_std": float(np.std([r["reward"] for r in all_rows])),
            "clipped_at_max_rate": float(np.mean([bool(r.get("clipped_at_max")) for r in all_rows])),
            "completion_tokens_mean": float(np.mean([r.get("completion_tokens", 0) for r in all_rows])),
        },
        "prompt_bins": {pid: {"bin": bins[pid], "mean_reward": prompt_means[pid]} for pid in by_prompt},
        "by_k": {},
    }
    table = []
    for k in cfg["group_sizes"]:
        k = int(k)
        groups = regroup_equal_generation_budget(by_prompt, k)
        overall = group_metrics(groups)
        subsets, _ = all_subsets_estimate(by_prompt, k)
        per_bin = {}
        for b in ["hard", "medium", "easy"]:
            gb = [g for g in groups if bins[g["prompt"]] == b]
            if gb:
                per_bin[b] = group_metrics(gb)
                sub_b, _ = all_subsets_estimate({p: v for p, v in by_prompt.items() if bins[p] == b}, k)
                per_bin[b]["all_subsets"] = sub_b
        out["by_k"][str(k)] = {"overall": overall, "all_subsets": subsets, "by_difficulty": per_bin}
        table.append({"K": k, "bin": "all", **overall, **{f"subsets_{kk}": v for kk, v in subsets.items()}})
        for b, m in per_bin.items():
            table.append({"K": k, "bin": b, **{kk: v for kk, v in m.items() if kk != "all_subsets"},
                          **{f"subsets_{kk}": v for kk, v in m["all_subsets"].items()}})
    save_json(results / "group_size_study.json", out)
    df = pd.DataFrame(table)
    df.to_csv(results / "group_size_summary.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df)


if __name__ == "__main__":
    main()
