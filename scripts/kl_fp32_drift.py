"""Float32 drift-from-reference for every trained/supplied adapter (Tasks 1-3).

Why: the released config runs models in float16, and scripts/kl_floor_check.py showed that float16
exact KL has a precision floor of ~1.1e-5 nats/token, which swamps the true drift of the PPO/GRPO
midpoints and their short continuations (true values ~2e-7 to 5e-7). The held-out KL columns of
Tasks 2-3 are therefore below resolution. This script re-measures drift in float32.

Protocol (identical for every adapter): the same N fixed responses (the first N held-out
generations of the GRPO midpoint, results/task3_grpo/generations_midpoint.jsonl) are
teacher-forced through each policy and through the reference (adapter disabled); we report the
exact full-vocabulary KL(pi || pi_ref) averaged over response tokens, and its standard error across
responses. Because every adapter is scored on the same tokens, differences between adapters are
paired. It is a drift measure on a fixed text set, not an on-policy KL.

    python -m scripts.kl_fp32_drift --n 48
Writes results/kl_fp32_drift.{json,csv}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.models import load_tokenizer
from common.policy_eval import exact_token_kl
from scripts.kl_floor_check import DEV, build_examples, logits_for


def collect_adapters() -> dict[str, str]:
    found = {}
    fixed = {
        "ppo_midpoint": "checkpoints/ppo_midpoint_policy",
        "grpo_midpoint": "checkpoints/grpo_midpoint_policy",
    }
    for k, v in fixed.items():
        found[k] = v
    for task, root in [("task1", "outputs/task1_dpo"), ("task2", "outputs/task2_ppo"), ("task3", "outputs/task3_grpo")]:
        base = repo_path(root)
        if not base.exists():
            continue
        for f in sorted(base.rglob("adapter_model.safetensors")):
            d = f.parent
            rel = d.relative_to(base).as_posix()
            if "critic" in rel or "value" in rel:
                continue  # critic, not a policy
            found[f"{task}/{rel}"] = str(d.relative_to(repo_path(".")))
    return {k: v for k, v in found.items() if (repo_path(v) / "adapter_model.safetensors").exists()}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--max-resp", type=int, default=512)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    examples = build_examples(cfg, tok, args.n, args.max_resp)
    adapters = collect_adapters()
    print(f"{len(adapters)} adapters, {len(examples)} responses:", *adapters, sep="\n  ", flush=True)

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.float32, low_cpu_mem_usage=True)
    names = list(adapters)
    model = PeftModel.from_pretrained(base, str(repo_path(adapters[names[0]])), adapter_name="a0")
    alias = {names[0]: "a0"}
    for i, nm in enumerate(names[1:], 1):
        alias[nm] = f"a{i}"
        model.load_adapter(str(repo_path(adapters[nm])), adapter_name=alias[nm])
    model = model.to(DEV).eval()

    per_resp = {nm: [] for nm in names}
    for j, (ids, plen, rlen) in enumerate(examples):
        with model.disable_adapter():
            ref = logits_for(model, ids, plen, rlen)
        for nm in names:
            model.set_adapter(alias[nm])
            pol = logits_for(model, ids, plen, rlen)
            per_resp[nm].append((exact_token_kl(pol, ref).sum().item(), rlen))
            del pol
        del ref
        print(f"  response {j + 1}/{len(examples)}", flush=True)

    rows = []
    for nm in names:
        kl = np.array([k for k, _ in per_resp[nm]])
        n = np.array([t for _, t in per_resp[nm]])
        per_tok = kl / n
        rows.append({
            "adapter": nm, "path": adapters[nm],
            "kl_per_token_fp32": float(kl.sum() / n.sum()),
            "kl_per_token_se": float(per_tok.std(ddof=1) / np.sqrt(len(per_tok))),
            "kl_per_response_mean_fp32": float(kl.mean()),
        })
    df = pd.DataFrame(rows)
    # paired difference vs the matching midpoint (Tasks 2/3)
    for i, r in df.iterrows():
        mid = "ppo_midpoint" if r["adapter"].startswith("task2/") else "grpo_midpoint" if r["adapter"].startswith("task3/") else None
        if mid:
            d = np.array([a / t - b / t for (a, t), (b, _) in zip(per_resp[r["adapter"]], per_resp[mid])])
            df.loc[i, "diff_vs_midpoint"] = d.mean()
            df.loc[i, "diff_vs_midpoint_se"] = d.std(ddof=1) / np.sqrt(len(d))
    out = repo_path("results")
    df.to_csv(out / "kl_fp32_drift.csv", index=False)
    save_json(out / "kl_fp32_drift.json", {"n_responses": len(examples), "rows": df.to_dict(orient="records")})
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.max_colwidth", 40):
        print(df.drop(columns=["path"]).to_string(index=False, float_format=lambda x: f"{x:.3e}"))


if __name__ == "__main__":
    main()
