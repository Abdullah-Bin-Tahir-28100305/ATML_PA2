"""Task 4: switch the manual audit from the 60-row 'rotate' sheet to the 240-row 'all_policies' sheet.

The 60 fixed audit prompts (30 SAFE + 30 UNSAFE, chosen by the course seed) are unchanged. The
240-row sheet pairs every audit prompt with all four policies' responses, so every policy is audited
on the same prompts.

What this script does (your existing labels are never lost or changed):
  1. Backs up the current results/task4_safety/manual_audit_labels.csv (the 60 rotate labels) to
     manual_audit_labels_rotate60.csv (only the first time; later runs reuse that backup).
  2. Builds the blinded 240-row sheet with task4_safety.make_audit_sheet --mode all_policies.
  3. Copies each of your 60 existing labels/notes onto the matching (xstest_id, policy) row.
  4. Optionally (on by default) copies a label to rows whose prompt AND response text are identical
     to a row you already labelled: the same prompt-response pair must receive the same label.
     These rows are marked in `prefill_source` so you can review them. Use --no-copy-identical to
     label them yourself instead.
  5. Writes results/task4_safety/manual_audit_sheet_240.csv. Fill the remaining blank
     `manual_label` cells, save the sheet as results/task4_safety/manual_audit_labels.csv, then run
     task4_safety.evaluate_safety.

    python -m task4_safety.expand_audit_240 --config configs/feedback.yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import pandas as pd

from common.data import load_yaml, repo_path

LABELS = {"SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--no-copy-identical", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"

    current = outdir / "manual_audit_labels.csv"
    backup = outdir / "manual_audit_labels_rotate60.csv"
    if not backup.exists():
        if not current.exists():
            raise SystemExit(f"No existing labels found at {current}")
        shutil.copy2(current, backup)
        print("Backed up your 60 rotate labels to", backup)
    else:
        print("Using existing backup of your 60 rotate labels:", backup)

    old = pd.read_csv(backup)
    old["manual_label"] = old["manual_label"].astype(str).str.strip().str.upper()
    old = old[old["manual_label"].isin(LABELS)]
    if "manual_note" not in old.columns:
        old["manual_note"] = ""
    old_map = {(int(r.xstest_id), str(r.policy)): (r.manual_label, r.manual_note) for r in old.itertuples()}
    print(f"Valid existing labels: {len(old_map)}")

    subprocess.run([sys.executable, "-m", "task4_safety.make_audit_sheet", "--config", args.config,
                    "--mode", "all_policies"], check=True)
    sheet = pd.read_csv(outdir / "manual_audit_sheet.csv")
    sheet["manual_label"] = ""
    sheet["manual_note"] = sheet["manual_note"].fillna("").astype(str)
    sheet["prefill_source"] = ""

    n_own = 0
    for i, r in sheet.iterrows():
        key = (int(r.xstest_id), str(r.policy))
        if key in old_map:
            lab, note = old_map[key]
            sheet.at[i, "manual_label"] = lab
            sheet.at[i, "manual_note"] = "" if pd.isna(note) else str(note)
            sheet.at[i, "prefill_source"] = "your rotate label"
            n_own += 1

    n_copy = 0
    if not args.no_copy_identical:
        own = sheet[sheet["prefill_source"] == "your rotate label"]
        for i, r in sheet.iterrows():
            if sheet.at[i, "manual_label"]:
                continue
            match = own[(own.xstest_id == r.xstest_id) & (own.response == r.response)]
            if len(match):
                m = match.iloc[0]
                sheet.at[i, "manual_label"] = m.manual_label
                sheet.at[i, "prefill_source"] = f"identical response to your {m.policy} label"
                n_copy += 1

    out = outdir / "manual_audit_sheet_240.csv"
    sheet.to_csv(out, index=False)
    remaining = int((sheet["manual_label"] == "").sum())
    print(f"\nWrote {out}  ({len(sheet)} rows)")
    print(f"  pre-filled from your 60 labels : {n_own}")
    print(f"  copied (identical response)    : {n_copy}")
    print(f"  left for you to label          : {remaining}")
    print("Label the blank rows, save as results/task4_safety/manual_audit_labels.csv, then run evaluate_safety.")


if __name__ == "__main__":
    main()
