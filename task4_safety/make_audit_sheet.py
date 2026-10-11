"""Task 4 Step 3 helper: build the blinded manual-audit sheet.

The 60 audit prompts (30 SAFE + 30 UNSAFE XSTest ids) are fixed by the course seed
(`fixed_audit_ids`). The sheet joins those ids to the generated responses WITHOUT the AI judge
labels, in a seeded random order, so labels can be assigned blind.

  --mode all_policies (default): every audit prompt x every policy (60 x 4 = 240 rows).
  --mode rotate: each audit prompt is paired with exactly one policy, rotating SFT/DPO/PPO/GRPO
                 inside each class (60 rows, 15 per policy, balanced safe/unsafe).

Fill the `manual_label` column with one of
SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL, AMBIGUOUS
(and optionally `manual_note`), save it as results/task4_safety/manual_audit_labels.csv, then run
task4_safety.evaluate_safety.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

POLICIES = ["sft", "dpo", "ppo", "grpo"]


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--mode", choices=["all_policies", "rotate"], default="all_policies")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    base_rows = read_jsonl(src)
    ids = fixed_audit_ids(base_rows, int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    gens = {}
    for p in POLICIES:
        f = outdir / f"generated_{p}.jsonl"
        if f.exists():
            gens[p] = {r["xstest_id"]: r for r in read_jsonl(f)}
    missing = [p for p in POLICIES if p not in gens]
    if missing:
        print("WARNING: responses missing for", missing, "- the sheet only covers available policies.")

    meta = {r["xstest_id"]: r for r in base_rows}
    pairs = []
    if args.mode == "all_policies":
        pairs = [(i, p) for i in ids for p in gens]
    else:
        avail = [p for p in POLICIES if p in gens]
        for cls in ["SAFE", "UNSAFE"]:
            cls_ids = [i for i in ids if meta[i]["benchmark_class"] == cls]
            pairs += [(i, avail[j % len(avail)]) for j, i in enumerate(cls_ids)]

    rows = []
    for i, p in pairs:
        g = gens[p][i]
        rows.append({
            "audit_row": None,
            "xstest_id": i,
            "policy": p,
            "benchmark_class": g["benchmark_class"],
            "type": g["type"],
            "prompt": g["prompt"],
            "response": g["response"],
            "manual_label": "",
            "manual_note": "",
        })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=int(cfg["seed"])).reset_index(drop=True)
    df["audit_row"] = np.arange(1, len(df) + 1)
    sheet = outdir / "manual_audit_sheet.csv"
    df.to_csv(sheet, index=False)
    print(f"Wrote blinded audit sheet with {len(df)} rows ({args.mode}): {sheet}")
    print("Label it without opening judged_*.jsonl, save as results/task4_safety/manual_audit_labels.csv,")
    print("then run: python -m task4_safety.evaluate_safety --config configs/feedback.yaml")


if __name__ == "__main__":
    main()
